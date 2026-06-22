from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_v5.parquet"
)

df = pd.read_parquet(FILE)

print("\nShape")
print(df.shape)

print("\nTarget Distribution")
print(
    df["target"]
    .value_counts()
)

print("\nFeature Nulls")

features = [
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

print(
    df[features]
    .isna()
    .sum()
)