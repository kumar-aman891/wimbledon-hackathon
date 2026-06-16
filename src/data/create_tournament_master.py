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
    / "tournament_master.parquet"
)


def create_tournament_master():

    df = pd.read_parquet(INPUT_FILE)

    tournament_master = (
        df[
            [
                "tourney_id",
                "tourney_name",
                "tourney_level"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "tourney_name",
                "tourney_id"
            ]
        )
    )

    tournament_master.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        tournament_master.head()
    )

    print(
        f"\nTournaments: {len(tournament_master)}"
    )


if __name__ == "__main__":

    create_tournament_master()