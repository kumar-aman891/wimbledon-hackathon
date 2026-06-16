import pandas as pd

matches = pd.read_parquet(
    "data/processed/unified_match_fact.parquet"
)

features = pd.read_parquet(
    "data/processed/match_features.parquet"
)

print("Matches :", len(matches))
print("Features:", len(features))
print("Difference:", len(features)-len(matches))