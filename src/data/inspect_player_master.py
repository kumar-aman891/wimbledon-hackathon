from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_master.parquet"
)

df = pd.read_parquet(FILE)

print("\nShape")
print(df.shape)

print("\nTour Distribution")
print(df["tour"].value_counts())

print("\nTop Players By Wins")
print(
    df[
        [
            "player_name",
            "tour",
            "career_grass_matches",
            "career_grass_wins",
            "career_grass_win_pct"
        ]
    ]
    .sort_values(
        "career_grass_wins",
        ascending=False
    )
    .head(20)
)