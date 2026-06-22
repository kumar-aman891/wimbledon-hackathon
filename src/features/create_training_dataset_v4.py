from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_v4.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_v4.parquet"
)


def create_training_dataset():

    df = pd.read_parquet(INPUT_FILE)

    winners = df.copy()
    winners["target"] = 1

    losers = df.copy()

    swap_pairs = [
        ("winner_id", "loser_id"),
        ("winner_name", "loser_name"),
        ("winner_rank", "loser_rank"),
        ("winner_key", "loser_key")
    ]

    for left, right in swap_pairs:
        losers[left], losers[right] = (
            losers[right],
            losers[left]
        )

    diff_features = [
        "elo_diff",
        "grass_elo_diff",
        "rank_diff",
        "career_win_pct_diff",
        "grass_win_pct_diff",
        "form_diff_5",
        "form_diff_10",
        "grass_form_diff_10",
        "h2h_win_pct_diff",
        "grass_h2h_win_pct_diff"
    ]

    for col in diff_features:
        losers[col] = losers[col] * -1

    losers["target"] = 0

    training = pd.concat(
        [winners, losers],
        ignore_index=True
    )

    training.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nTraining Dataset V4")
    print(training.shape)

    print("\nTarget Distribution")
    print(
        training["target"]
        .value_counts()
    )

    return training


if __name__ == "__main__":
    create_training_dataset()