from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_match_fact.parquet"
)

df = pd.read_parquet(FILE)

print("\nShape")
print(df.shape)

print("\nTournament Levels")
print(df["tourney_level"].value_counts())

print("\nSurfaces")
print(df["surface"].value_counts())

print("\nTours")
print(df["tour"].value_counts())

print("\nNull Counts")
print(
    df.isnull()
      .sum()
      .sort_values(ascending=False)
      .head(20)
)