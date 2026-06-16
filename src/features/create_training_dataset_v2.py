from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_v3.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_v2.parquet"
)


def create_training_dataset_v2():

    df = pd.read_parquet(INPUT_FILE)

    features = [
        "elo_diff",
        "grass_elo_diff",
        "rank_diff",
        "career_win_pct_diff",
        "grass_win_pct_diff",
        "form_diff_5",
        "form_diff_10",
        "grass_form_diff_10"
    ]

    winner_df = df[features].copy()

    winner_df["target"] = 1

    loser_df = pd.DataFrame()

    for col in features:

        loser_df[col] = -df[col]

    loser_df["target"] = 0

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

    print("\nTraining Dataset V2")

    print(training.shape)

    print(
        "\nTarget Distribution"
    )

    print(
        training["target"]
        .value_counts()
    )

    return training


if __name__ == "__main__":
    create_training_dataset_v2()