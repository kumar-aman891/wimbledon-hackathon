from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "head_to_head.parquet"
)

df = pd.read_parquet(FILE)

print(df.shape)

print(df.head())

print(
    "\nDuplicate Keys:",
    df.duplicated(
        [
            "player_key",
            "opponent_key"
        ]
    ).sum()
)