from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features.parquet"
)

df = pd.read_parquet(FILE)

cols = [
    "elo_diff",
    "grass_elo_diff",
    "rank_diff",
    "career_win_pct_diff",
    "grass_win_pct_diff",
    "target"
]

print(df[cols].head(20))

print(df[cols].describe())