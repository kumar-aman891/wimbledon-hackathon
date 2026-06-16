from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_match_fact.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "surface_master.parquet"
)


def create_surface_master():

    df = pd.read_parquet(INPUT_FILE)

    df["surface"] = (
        df["surface"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    surface_master = pd.DataFrame(
        {
            "surface":
            sorted(
                df["surface"]
                .dropna()
                .unique()
            )
        }
    )

    surface_master.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(surface_master)


if __name__ == "__main__":

    create_surface_master()