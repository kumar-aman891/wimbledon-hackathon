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
    / "matches_for_elo.parquet"
)


def prepare_matches():

    df = pd.read_parquet(INPUT_FILE)

    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"],
        format="%Y%m%d"
    )

    df = df.sort_values(
        [
            "tourney_date",
            "match_num"
        ]
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(df.shape)

    print(
        df[
            [
                "tourney_date",
                "winner_name",
                "loser_name"
            ]
        ].head()
    )


if __name__ == "__main__":
    prepare_matches()