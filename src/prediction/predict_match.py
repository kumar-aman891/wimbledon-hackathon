from pathlib import Path
import argparse

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


############################################################
# Files
############################################################

SNAPSHOT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_snapshot.parquet"
)

H2H_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "head_to_head_v2.parquet"
)

GRASS_H2H_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "grass_head_to_head_v2.parquet"
)

WINNER_MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "catboost_final.pkl"
)

MARGIN_MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "catboost_margin.pkl"
)


############################################################
# Load Resources
############################################################

print("Loading models and datasets...")

snapshot = pd.read_parquet(
    SNAPSHOT_FILE
)

h2h_df = pd.read_parquet(
    H2H_FILE
)

grass_h2h_df = pd.read_parquet(
    GRASS_H2H_FILE
)

winner_model = joblib.load(
    WINNER_MODEL_FILE
)

margin_model = joblib.load(
    MARGIN_MODEL_FILE
)


############################################################
# Feature List
############################################################

FEATURE_COLUMNS = [

    "elo_diff",
    "grass_elo_diff",
    "rank_diff",
    "career_win_pct_diff",
    "grass_win_pct_diff",
    "experience_diff",
    "form_diff_5",
    "form_diff_10",
    "grass_form_diff_10",
    "h2h_win_pct_diff",
    "grass_h2h_win_pct_diff",
    "age_diff",
    "grass_specialization_diff",

    "is_grand_slam",
    "is_masters",
    "is_atp500",
    "is_atp250",

    "round_BR",
    "round_F",
    "round_QF",
    "round_R128",
    "round_R16",
    "round_R32",
    "round_R64",
    "round_RR",
    "round_SF",

    "surface_Carpet",
    "surface_Clay",
    "surface_Grass",
    "surface_Hard",
    "surface_clay"

]

############################################################
# Helper Functions
############################################################

def get_player(player_name):
    """
    Return player snapshot row.

    Raises
    ------
    ValueError
        If player not found.
    """

    player = snapshot[
        snapshot["player_name"].str.lower()
        ==
        player_name.lower()
    ]

    if len(player) == 0:

        available = (
            snapshot["player_name"]
            .sort_values()
            .tolist()
        )

        raise ValueError(
            f"\nPlayer '{player_name}' not found.\n\n"
            f"Example names:\n"
            + "\n".join(available[:20])
        )

    return player.iloc[0]


def get_latest_h2h(
    player_key,
    opponent_key
):
    """
    Returns latest H2H win percentage.
    """

    rows = h2h_df[

        (h2h_df["player_key"] == player_key)
        &
        (h2h_df["opponent_key"] == opponent_key)

    ]

    if rows.empty:
        return 50.0

    rows = (
        rows
        .sort_values("match_key")
        .tail(1)
    )

    return float(
        rows.iloc[0]["h2h_win_pct_before"]
    )


def get_latest_grass_h2h(
    player_key,
    opponent_key
):
    """
    Returns latest Grass H2H win percentage.
    """

    rows = grass_h2h_df[

        (grass_h2h_df["player_key"] == player_key)
        &
        (grass_h2h_df["opponent_key"] == opponent_key)

    ]

    if rows.empty:
        return 50.0

    rows = (
        rows
        .sort_values("match_key")
        .tail(1)
    )

    return float(
        rows.iloc[0][
            "grass_h2h_win_pct_before"
        ]
    )


def print_player_summary(player):

    print()

    print("----------------------------------------")

    print(player["player_name"])

    print("----------------------------------------")

    print(
        f"Rank                : {player['rank']}"
    )

    print(
        f"Overall Elo         : {player['overall_elo']:.2f}"
    )

    print(
        f"Grass Elo           : {player['grass_elo']:.2f}"
    )

    print(
        f"Career Win %        : {player['career_win_pct']:.2f}"
    )

    print(
        f"Grass Win %         : {player['career_grass_win_pct']:.2f}"
    )

    print(
        f"Recent Form (10)    : {player['win_pct_last_10']:.2f}"
    )

    print(
        f"Grass Form (10)     : {player['grass_win_pct_last_10']:.2f}"
    )

    print(
        f"Career Matches      : {int(player['career_matches'])}"
    )

    print(
        f"Age                 : {player['age']:.1f}"
    )

    print()


############################################################
# Tournament Encoding
############################################################

def tournament_flags(tournament):

    return {

        "is_grand_slam":
            int(tournament == "GRAND_SLAM"),

        "is_masters":
            int(tournament == "MASTERS"),

        "is_atp500":
            int(tournament == "ATP500"),

        "is_atp250":
            int(tournament == "ATP250")

    }


############################################################
# Round Encoding
############################################################

def round_flags(round_name):

    rounds = {

        "round_BR":0,
        "round_F":0,
        "round_QF":0,
        "round_R128":0,
        "round_R16":0,
        "round_R32":0,
        "round_R64":0,
        "round_RR":0,
        "round_SF":0

    }

    key = f"round_{round_name}"

    if key in rounds:

        rounds[key] = 1

    return rounds


############################################################
# Surface Encoding
############################################################

def surface_flags(surface):

    surfaces = {

        "surface_Carpet":0,
        "surface_Clay":0,
        "surface_Grass":0,
        "surface_Hard":0,
        "surface_clay":0

    }

    if surface.upper() == "GRASS":

        surfaces["surface_Grass"] = 1

    elif surface.upper() == "CLAY":

        surfaces["surface_Clay"] = 1

    elif surface.upper() == "HARD":

        surfaces["surface_Hard"] = 1

    return surfaces

############################################################
# Feature Engineering
############################################################

def build_features(
    p1,
    p2,
    surface,
    round_name,
    tournament
):

    features = {}

    ########################################################
    # Core Features
    ########################################################

    features["elo_diff"] = (
        p1["overall_elo"]
        - p2["overall_elo"]
    )

    features["grass_elo_diff"] = (
        p1["grass_elo"]
        - p2["grass_elo"]
    )

    features["rank_diff"] = (
        p2["rank"]
        - p1["rank"]
    )

    features["career_win_pct_diff"] = (
        p1["career_win_pct"]
        - p2["career_win_pct"]
    )

    features["grass_win_pct_diff"] = (
        p1["career_grass_win_pct"]
        - p2["career_grass_win_pct"]
    )

    features["experience_diff"] = (
        p1["career_matches"]
        - p2["career_matches"]
    )

    ########################################################
    # Recent Form
    ########################################################

    features["form_diff_5"] = (
        p1["win_pct_last_5"]
        - p2["win_pct_last_5"]
    )

    features["form_diff_10"] = (
        p1["win_pct_last_10"]
        - p2["win_pct_last_10"]
    )

    features["grass_form_diff_10"] = (
        p1["grass_win_pct_last_10"]
        - p2["grass_win_pct_last_10"]
    )

    ########################################################
    # Age
    ########################################################

    features["age_diff"] = (
        p1["age"]
        - p2["age"]
    )

    ########################################################
    # Grass Specialization
    ########################################################

    features["grass_specialization_diff"] = (
        p1["grass_specialization"]
        - p2["grass_specialization"]
    )

    ########################################################
    # Head-to-Head
    ########################################################

    p1_h2h = get_latest_h2h(
        p1["player_key"],
        p2["player_key"]
    )

    p2_h2h = get_latest_h2h(
        p2["player_key"],
        p1["player_key"]
    )

    features["h2h_win_pct_diff"] = (
        p1_h2h
        - p2_h2h
    )

    ########################################################
    # Grass Head-to-Head
    ########################################################

    p1_grass_h2h = get_latest_grass_h2h(
        p1["player_key"],
        p2["player_key"]
    )

    p2_grass_h2h = get_latest_grass_h2h(
        p2["player_key"],
        p1["player_key"]
    )

    features["grass_h2h_win_pct_diff"] = (
        p1_grass_h2h
        - p2_grass_h2h
    )

    ########################################################
    # Tournament
    ########################################################

    features.update(
        tournament_flags(
            tournament
        )
    )

    ########################################################
    # Round
    ########################################################

    features.update(
        round_flags(
            round_name
        )
    )

    ########################################################
    # Surface
    ########################################################

    features.update(
        surface_flags(
            surface
        )
    )

    ########################################################
    # Ensure every expected feature exists
    ########################################################

    for col in FEATURE_COLUMNS:

        if col not in features:

            features[col] = 0

    ########################################################
    # Return DataFrame
    ########################################################

    X = (
        pd.DataFrame([features])
        [FEATURE_COLUMNS]
    )

    return X


############################################################
# Explain Prediction
############################################################

def explain_prediction(X):

    important = [

        "career_win_pct_diff",
        "grass_win_pct_diff",
        "elo_diff",
        "grass_elo_diff",
        "rank_diff",
        "form_diff_10",
        "grass_form_diff_10",
        "h2h_win_pct_diff"

    ]

    print()
    print("=" * 60)
    print("Key Advantages")
    print("=" * 60)

    for feature in important:

        value = X.iloc[0][feature]

        if value > 0:

            print(
                f"+ {feature:<30} {value:8.2f}"
            )

        elif value < 0:

            print(
                f"- {feature:<30} {value:8.2f}"
            )

############################################################
# Prediction Engine
############################################################

def predict_match(
    player1_name,
    player2_name,
    surface,
    round_name,
    tournament
):

    ########################################################
    # Player Profiles
    ########################################################

    p1 = get_player(player1_name)
    p2 = get_player(player2_name)

    ########################################################
    # Build Features
    ########################################################

    X = build_features(

        p1,
        p2,

        surface,

        round_name,

        tournament

    )

    ########################################################
    # Winner Model
    ########################################################

    winner_probability = (
        winner_model
        .predict_proba(X)[0][1]
    )

    if winner_probability >= 0.5:

        winner = p1["player_name"]

        winner_probability_display = winner_probability

    else:

        winner = p2["player_name"]

        winner_probability_display = 1 - winner_probability

    ########################################################
    # Margin Model
    ########################################################

    margin_probability = (
        margin_model
        .predict_proba(X)[0][1]
    )

    if margin_probability >= 0.5:

        margin_prediction = "Greater than 1 Set"

    else:

        margin_prediction = "Exactly 1 Set"

    margin_confidence = max(

        margin_probability,

        1 - margin_probability

    )

    ########################################################
    # Print Result
    ########################################################

    print()

    print("=" * 70)

    print("WIMBLEDON MATCH PREDICTION")

    print("=" * 70)

    print()

    print("Player 1")

    print_player_summary(p1)

    print("Player 2")

    print_player_summary(p2)

    print("=" * 70)

    print("Prediction")

    print("=" * 70)

    print()

    print(f"Winner              : {winner}")

    print(
        f"Win Probability     : "
        f"{winner_probability_display:.1%}"
    )

    print()

    print(
        f"Winning Margin      : "
        f"{margin_prediction}"
    )

    print(
        f"Margin Confidence   : "
        f"{margin_confidence:.1%}"
    )

    ########################################################
    # Head-to-Head
    ########################################################

    p1_h2h = get_latest_h2h(

        p1["player_key"],

        p2["player_key"]

    )

    p2_h2h = get_latest_h2h(

        p2["player_key"],

        p1["player_key"]

    )

    print()

    print("=" * 70)

    print("Head-to-Head")

    print("=" * 70)

    print()

    print(

        f"{p1['player_name']} : "

        f"{p1_h2h:.1f}%"

    )

    print(

        f"{p2['player_name']} : "

        f"{p2_h2h:.1f}%"

    )

    ########################################################
    # Important Features
    ########################################################

    explain_prediction(X)

    ########################################################
    # Return Dictionary
    ########################################################

    return {

        "player1": p1["player_name"],

        "player2": p2["player_name"],

        "winner": winner,

        "winner_probability": winner_probability_display,

        "margin_prediction": margin_prediction,

        "margin_confidence": margin_confidence

    }

############################################################
# Main
############################################################

def main():

    parser = argparse.ArgumentParser(

        description="Wimbledon Match Predictor"

    )

    parser.add_argument(

        "--player1",

        required=True,

        help="Player 1"

    )

    parser.add_argument(

        "--player2",

        required=True,

        help="Player 2"

    )

    parser.add_argument(

        "--surface",

        default="GRASS",

        choices=[

            "GRASS",

            "CLAY",

            "HARD"

        ]

    )

    parser.add_argument(

        "--round",

        default="R128",

        choices=[

            "R128",

            "R64",

            "R32",

            "R16",

            "QF",

            "SF",

            "F"

        ]

    )

    parser.add_argument(

        "--tournament",

        default="GRAND_SLAM",

        choices=[

            "GRAND_SLAM",

            "MASTERS",

            "ATP500",

            "ATP250"

        ]

    )

    args = parser.parse_args()

    ########################################################

    if args.player1.lower() == args.player2.lower():

        raise ValueError(

            "Player 1 and Player 2 cannot be the same."

        )

    ########################################################

    try:

        result = predict_match(

            player1_name=args.player1,

            player2_name=args.player2,

            surface=args.surface,

            round_name=args.round,

            tournament=args.tournament

        )

    except Exception as e:

        print()

        print("=" * 70)

        print("Prediction Failed")

        print("=" * 70)

        print()

        print(e)

        print()

        raise

    ########################################################

    print()

    print("=" * 70)

    print("Prediction Complete")

    print("=" * 70)

    print()

    print(

        f"Winner : "

        f"{result['winner']}"

    )

    print(

        f"Win Probability : "

        f"{result['winner_probability']:.1%}"

    )

    print(

        f"Winning Margin : "

        f"{result['margin_prediction']}"

    )

    print(

        f"Margin Confidence : "

        f"{result['margin_confidence']:.1%}"

    )

    print()

    print("=" * 70)


############################################################

if __name__ == "__main__":

    main()