from pathlib import Path
import argparse
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "catboost_final.pkl"
)


def get_player(snapshot, player_name):

    player = snapshot[
        snapshot["player_name"]
        .str.lower()
        ==
        player_name.lower()
    ]

    if len(player) == 0:
        raise ValueError(
            f"Player not found: {player_name}"
        )

    return player.iloc[0]


def get_h2h_feature(
    h2h_df,
    player_key,
    opponent_key,
    column_name
):

    row = h2h_df[
        (h2h_df["player_key"] == player_key)
        &
        (h2h_df["opponent_key"] == opponent_key)
    ]

    if len(row) == 0:
        return 50.0

    row = (
        row
        .sort_values("match_key")
        .tail(1)
    )

    return float(
        row.iloc[0][column_name]
    )


def build_feature_vector(
    p1,
    p2,
    surface,
    round_name,
    tournament_type,
    h2h_df,
    grass_h2h_df
):

    features = {}

    # ----------------------------------
    # Core Features
    # ----------------------------------

    features["elo_diff"] = (
        p1["overall_elo"]
        -
        p2["overall_elo"]
    )

    features["grass_elo_diff"] = (
        p1["grass_elo"]
        -
        p2["grass_elo"]
    )

    features["rank_diff"] = (
        p2["rank"]
        -
        p1["rank"]
    )

    features["career_win_pct_diff"] = (
        p1["career_win_pct"]
        -
        p2["career_win_pct"]
    )

    features["grass_win_pct_diff"] = (
        p1["career_grass_win_pct"]
        -
        p2["career_grass_win_pct"]
    )

    features["experience_diff"] = (
        p1["career_matches"]
        -
        p2["career_matches"]
    )

    features["form_diff_5"] = (
        p1["win_pct_last_5"]
        -
        p2["win_pct_last_5"]
    )

    features["form_diff_10"] = (
        p1["win_pct_last_10"]
        -
        p2["win_pct_last_10"]
    )

    features["grass_form_diff_10"] = (
        p1["grass_win_pct_last_10"]
        -
        p2["grass_win_pct_last_10"]
    )

    features["age_diff"] = (
        p1["age"]
        -
        p2["age"]
    )

    features["grass_specialization_diff"] = (
        p1["grass_specialization"]
        -
        p2["grass_specialization"]
    )

    # ----------------------------------
    # H2H
    # ----------------------------------

    p1_h2h = get_h2h_feature(
        h2h_df,
        p1["player_key"],
        p2["player_key"],
        "h2h_win_pct_before"
    )

    p2_h2h = get_h2h_feature(
        h2h_df,
        p2["player_key"],
        p1["player_key"],
        "h2h_win_pct_before"
    )

    features["h2h_win_pct_diff"] = (
        p1_h2h
        -
        p2_h2h
    )

    # ----------------------------------
    # Grass H2H
    # ----------------------------------

    p1_grass_h2h = get_h2h_feature(
        grass_h2h_df,
        p1["player_key"],
        p2["player_key"],
        "grass_h2h_win_pct_before"
    )

    p2_grass_h2h = get_h2h_feature(
        grass_h2h_df,
        p2["player_key"],
        p1["player_key"],
        "grass_h2h_win_pct_before"
    )

    features["grass_h2h_win_pct_diff"] = (
        p1_grass_h2h
        -
        p2_grass_h2h
    )

    # ----------------------------------
    # Tournament
    # ----------------------------------

    features["is_grand_slam"] = int(
        tournament_type.upper()
        == "GRAND_SLAM"
    )

    features["is_masters"] = int(
        tournament_type.upper()
        == "MASTERS"
    )

    features["is_atp500"] = int(
        tournament_type.upper()
        == "ATP500"
    )

    features["is_atp250"] = int(
        tournament_type.upper()
        == "ATP250"
    )

    # ----------------------------------
    # Round
    # ----------------------------------

    round_cols = [
        "round_BR",
        "round_F",
        "round_QF",
        "round_R128",
        "round_R16",
        "round_R32",
        "round_R64",
        "round_RR",
        "round_SF"
    ]

    for col in round_cols:
        features[col] = 0

    round_map = {
        "BR": "round_BR",
        "F": "round_F",
        "QF": "round_QF",
        "R128": "round_R128",
        "R64": "round_R64",
        "R32": "round_R32",
        "R16": "round_R16",
        "RR": "round_RR",
        "SF": "round_SF"
    }

    if round_name in round_map:
        features[
            round_map[round_name]
        ] = 1

    # ----------------------------------
    # Surface
    # ----------------------------------

    surface_cols = [
        "surface_Carpet",
        "surface_Clay",
        "surface_Grass",
        "surface_Hard",
        "surface_clay"
    ]

    for col in surface_cols:
        features[col] = 0

    if surface.upper() == "GRASS":
        features["surface_Grass"] = 1

    elif surface.upper() == "CLAY":
        features["surface_Clay"] = 1

    elif surface.upper() == "HARD":
        features["surface_Hard"] = 1

    elif surface.upper() == "CARPET":
        features["surface_Carpet"] = 1

    return pd.DataFrame(
        [features]
    )


def predict_match():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--player1",
        required=True
    )

    parser.add_argument(
        "--player2",
        required=True
    )

    parser.add_argument(
        "--surface",
        default="GRASS"
    )

    parser.add_argument(
        "--round",
        default="R128"
    )

    parser.add_argument(
        "--tournament",
        default="GRAND_SLAM"
    )

    args = parser.parse_args()

    snapshot = pd.read_parquet(
        SNAPSHOT_FILE
    )

    h2h_df = pd.read_parquet(
        H2H_FILE
    )

    grass_h2h_df = pd.read_parquet(
        GRASS_H2H_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    p1 = get_player(
        snapshot,
        args.player1
    )

    p2 = get_player(
        snapshot,
        args.player2
    )

    X = build_feature_vector(
        p1,
        p2,
        args.surface,
        args.round,
        args.tournament,
        h2h_df,
        grass_h2h_df
    )

    probability = (
        model.predict_proba(X)
        [0][1]
    )

    print("\nMatch Prediction")
    print("-" * 60)

    print(
        f"{p1['player_name']} : {probability:.1%}"
    )

    print(
        f"{p2['player_name']} : {(1-probability):.1%}"
    )

    print("\nKey Advantages")

    key_features = [
        "career_win_pct_diff",
        "grass_win_pct_diff",
        "elo_diff",
        "grass_elo_diff",
        "form_diff_10",
        "h2h_win_pct_diff"
    ]

    for feature in key_features:

        value = X.iloc[0][feature]

        if value > 0:
            print(
                f"+ {feature}: {value:.2f}"
            )

        elif value < 0:
            print(
                f"- {feature}: {value:.2f}"
            )


if __name__ == "__main__":
    predict_match()