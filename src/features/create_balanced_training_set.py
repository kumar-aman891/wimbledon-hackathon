from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset.parquet"
)


def create_training_dataset():

    df = pd.read_parquet(INPUT_FILE)

    feature_cols = [
        "elo_diff",
        "grass_elo_diff",
        "rank_diff",
        "career_win_pct_diff",
        "grass_win_pct_diff"
    ]

    # ------------------------
    # Winner Perspective
    # ------------------------

    winner_df = df[feature_cols].copy()

    winner_df["target"] = 1

    # ------------------------
    # Loser Perspective
    # ------------------------

    loser_df = pd.DataFrame()

    loser_df["elo_diff"] = (
        -df["elo_diff"]
    )

    loser_df["grass_elo_diff"] = (
        -df["grass_elo_diff"]
    )

    loser_df["rank_diff"] = (
        -df["rank_diff"]
    )

    loser_df["career_win_pct_diff"] = (
        -df["career_win_pct_diff"]
    )

    loser_df["grass_win_pct_diff"] = (
        -df["grass_win_pct_diff"]
    )

    loser_df["target"] = 0

    # ------------------------
    # Combine
    # ------------------------

    training = pd.concat(
        [
            winner_df,
            loser_df
        ],
        ignore_index=True
    )

    training.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nTraining Dataset")

    print(training.shape)

    print("\nTarget Distribution")

    print(
        training["target"]
        .value_counts()
    )

    return training


if __name__ == "__main__":
    create_training_dataset()