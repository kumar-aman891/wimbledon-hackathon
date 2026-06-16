from pathlib import Path
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_match_fact.parquet"
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
        matches["tourney_date"],
        format="%Y%m%d",
        errors="coerce"
    )

    matches = matches.sort_values(
        "tourney_date"
    )

    player_history = defaultdict(list)
    grass_history = defaultdict(list)

    records = []

    for _, row in matches.iterrows():

        winner_id = row["winner_id"]
        loser_id = row["loser_id"]

        tour = row["tour"]

        surface = str(
            row["surface"]
        ).upper()

        match_key = (
            str(row["tourney_id"])
            + "_"
            + str(row["match_num"])
        )

        winner_hist = player_history[winner_id]
        loser_hist = player_history[loser_id]

        winner_last_5 = winner_hist[-5:]
        winner_last_10 = winner_hist[-10:]

        loser_last_5 = loser_hist[-5:]
        loser_last_10 = loser_hist[-10:]

        winner_grass_last_10 = (
            grass_history[winner_id][-10:]
        )

        loser_grass_last_10 = (
            grass_history[loser_id][-10:]
        )

        # Winner snapshot

        records.append(
            {
                "match_key": match_key,
                "player_id": winner_id,
                "tour": tour,
                "match_date": row["tourney_date"],

                "wins_last_5":
                    sum(winner_last_5),

                "matches_last_5":
                    len(winner_last_5),

                "win_pct_last_5":
                    (
                        sum(winner_last_5)
                        /
                        len(winner_last_5)
                        * 100
                    )
                    if len(winner_last_5) > 0
                    else 0,

                "wins_last_10":
                    sum(winner_last_10),

                "matches_last_10":
                    len(winner_last_10),

                "win_pct_last_10":
                    (
                        sum(winner_last_10)
                        /
                        len(winner_last_10)
                        * 100
                    )
                    if len(winner_last_10) > 0
                    else 0,

                "grass_wins_last_10":
                    sum(winner_grass_last_10),

                "grass_matches_last_10":
                    len(winner_grass_last_10),

                "grass_win_pct_last_10":
                    (
                        sum(winner_grass_last_10)
                        /
                        len(winner_grass_last_10)
                        * 100
                    )
                    if len(winner_grass_last_10) > 0
                    else 0
            }
        )

        # Loser snapshot

        records.append(
            {
                "match_key": match_key,
                "player_id": loser_id,
                "tour": tour,
                "match_date": row["tourney_date"],

                "wins_last_5":
                    sum(loser_last_5),

                "matches_last_5":
                    len(loser_last_5),

                "win_pct_last_5":
                    (
                        sum(loser_last_5)
                        /
                        len(loser_last_5)
                        * 100
                    )
                    if len(loser_last_5) > 0
                    else 0,

                "wins_last_10":
                    sum(loser_last_10),

                "matches_last_10":
                    len(loser_last_10),

                "win_pct_last_10":
                    (
                        sum(loser_last_10)
                        /
                        len(loser_last_10)
                        * 100
                    )
                    if len(loser_last_10) > 0
                    else 0,

                "grass_wins_last_10":
                    sum(loser_grass_last_10),

                "grass_matches_last_10":
                    len(loser_grass_last_10),

                "grass_win_pct_last_10":
                    (
                        sum(loser_grass_last_10)
                        /
                        len(loser_grass_last_10)
                        * 100
                    )
                    if len(loser_grass_last_10) > 0
                    else 0
            }
        )

        # Update histories AFTER snapshot

        player_history[winner_id].append(1)
        player_history[loser_id].append(0)

        if surface == "GRASS":

            grass_history[winner_id].append(1)
            grass_history[loser_id].append(0)

    recent_form = pd.DataFrame(records)

    recent_form.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nRecent Form Dataset")
    print(recent_form.shape)

    print(
        "\nDuplicate Keys:",
        recent_form[
            ["match_key", "player_id"]
        ]
        .duplicated()
        .sum()
    )

    return recent_form


if __name__ == "__main__":
    build_recent_form()