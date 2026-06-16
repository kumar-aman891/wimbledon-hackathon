from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recent_form.parquet"
)

df = pd.read_parquet(FILE)

print(df.head(20))

print(df.describe())