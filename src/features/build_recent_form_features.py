from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches_for_elo.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recent_form.parquet"
)


def build_recent_form():

    matches = pd.read_parquet(MATCH_FILE)

    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"]
    )

    records = []

    player_history = {}

    for _, row in matches.iterrows():

        match_date = row["tourney_date"]

        winner = row["winner_id"]
        loser = row["loser_id"]

        winner_hist = player_history.get(
            winner,
            []
        )

        loser_hist = player_history.get(
            loser,
            []
        )

        winner_last_5 = sum(
            winner_hist[-5:]
        )

        winner_last_10 = sum(
            winner_hist[-10:]
        )

        loser_last_5 = sum(
            loser_hist[-5:]
        )

        loser_last_10 = sum(
            loser_hist[-10:]
        )

        records.append(
            {
                "match_date": match_date,
                "winner_id": winner,
                "loser_id": loser,

                "winner_wins_last_5":
                    winner_last_5,

                "winner_wins_last_10":
                    winner_last_10,

                "loser_wins_last_5":
                    loser_last_5,

                "loser_wins_last_10":
                    loser_last_10
            }
        )

        player_history.setdefault(
            winner,
            []
        ).append(1)

        player_history.setdefault(
            loser,
            []
        ).append(0)

    recent_form = pd.DataFrame(
        records
    )

    recent_form.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        recent_form.shape
    )

    return recent_form


if __name__ == "__main__":
    build_recent_form()