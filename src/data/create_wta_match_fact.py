from pathlib import Path

from src.data.load_matches import (
    load_matches
)
from src.data.schema_standardizer import (
    standardize_schema
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    exist_ok=True,
    parents=True
)


df = load_matches(
    "wta",
    2015,
    2025
)

df = standardize_schema(df)

df.to_parquet(
    OUTPUT_DIR
    / "wta_match_fact.parquet",
    index=False
)

print(df.shape)