from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw"


def load_matches(
        tour: str,
        start_year: int,
        end_year: int
):

    tour = tour.lower()

    dfs = []

    for year in range(
            start_year,
            end_year + 1
    ):

        file_path = (
            RAW_DATA
            / tour
            / f"{tour}_matches_{year}.csv"
        )

        print(
            f"Loading {file_path.name}"
        )

        df = pd.read_csv(
            file_path
        )

        df["match_year"] = year
        df["tour"] = tour.upper()

        dfs.append(df)

    return pd.concat(
        dfs,
        ignore_index=True
    )