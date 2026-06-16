from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_match_fact.parquet"
)

PLAYER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_master.parquet"
)

ELO_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_elo.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features.parquet"
)


def build_features():

    print("Loading matches...")
    matches = pd.read_parquet(MATCH_FILE)

    # ----------------------------------
    # Create ATP/WTA-safe keys
    # ----------------------------------

    matches["winner_key"] = (
        matches["tour"]
        + "_"
        + matches["winner_id"].astype(str)
    )

    matches["loser_key"] = (
        matches["tour"]
        + "_"
        + matches["loser_id"].astype(str)
    )

    print("Loading player master...")
    players = pd.read_parquet(PLAYER_FILE)

    print("Loading elo ratings...")
    elo = pd.read_parquet(ELO_FILE)

    # ----------------------------------
    # Player Features
    # ----------------------------------

    player_features = (
        players[
            [
                "player_key",
                "player_id",
                "tour",
                "career_matches",
                "career_wins",
                "career_losses",
                "career_win_pct",
                "career_grass_matches",
                "career_grass_wins",
                "career_grass_losses",
                "career_grass_win_pct"
            ]
        ]
        .merge(
            elo,
            on="player_id",
            how="left"
        )
    )

    # ----------------------------------
    # Winner Features
    # ----------------------------------

    winner_features = (
        player_features
        .add_prefix("winner_")
    )

    matches = matches.merge(
        winner_features,
        left_on="winner_key",
        right_on="winner_player_key",
        how="left"
    )

    # ----------------------------------
    # Loser Features
    # ----------------------------------

    loser_features = (
        player_features
        .add_prefix("loser_")
    )

    matches = matches.merge(
        loser_features,
        left_on="loser_key",
        right_on="loser_player_key",
        how="left"
    )

    # ----------------------------------
    # Derived Features
    # ----------------------------------

    matches["elo_diff"] = (
        matches["winner_overall_elo"]
        -
        matches["loser_overall_elo"]
    )

    matches["grass_elo_diff"] = (
        matches["winner_grass_elo"]
        -
        matches["loser_grass_elo"]
    )

    matches["rank_diff"] = (
        matches["loser_rank"]
        -
        matches["winner_rank"]
    )

    matches["career_win_pct_diff"] = (
        matches["winner_career_win_pct"]
        -
        matches["loser_career_win_pct"]
    )

    matches["grass_win_pct_diff"] = (
        matches["winner_career_grass_win_pct"]
        -
        matches["loser_career_grass_win_pct"]
    )

    # ----------------------------------
    # Target
    # ----------------------------------

    matches["target"] = 1

    # ----------------------------------
    # Save
    # ----------------------------------

    matches.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature Store Created")

    print(
        f"Rows: {len(matches):,}"
    )

    print(
        f"Columns: {len(matches.columns)}"
    )

    print("\nNull Check")

    print(
        matches[
            [
                "winner_overall_elo",
                "loser_overall_elo",
                "winner_grass_elo",
                "loser_grass_elo"
            ]
        ]
        .isna()
        .sum()
    )

    return matches


if __name__ == "__main__":
    build_features()