from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

atp = pd.read_parquet(
    PROCESSED
    / "atp_match_fact.parquet"
)

wta = pd.read_parquet(
    PROCESSED
    / "wta_match_fact.parquet"
)

unified = pd.concat(
    [atp, wta],
    ignore_index=True
)

print("ATP:", atp.shape)
print("WTA:", wta.shape)
print("Unified:", unified.shape)

unified.to_parquet(
    PROCESSED
    / "unified_match_fact.parquet",
    index=False
)