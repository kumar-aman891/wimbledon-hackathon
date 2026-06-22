from pathlib import Path

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
    / "grass_head_to_head_v2.parquet"
)


def build_grass_head_to_head_v2():

    matches = pd.read_parquet(
        MATCH_FILE
    )

    # ----------------------------------
    # Grass Only
    # ----------------------------------

    matches = matches[
        matches["surface"]
        .str.upper()
        == "GRASS"
    ].copy()

    # ----------------------------------
    # Keys
    # ----------------------------------

    matches["match_key"] = (
        matches["tourney_id"].astype(str)
        + "_"
        + matches["match_num"].astype(str)
    )

    matches["winner_key"] = (
        matches["tour"]
        + "_"
        + matches["winner_id"].astype(str)
    )

    matches["loser_key"] = (
        matches["tour"]
        + "_"
        + matches["loser_id"].astype(str)
    )

    matches["match_date"] = pd.to_datetime(
        matches["tourney_date"]
    )

    matches = matches.sort_values(
        [
            "match_date",
            "match_num"
        ]
    )

    # ----------------------------------
    # Tracker
    # ----------------------------------

    tracker = {}

    rows = []

    total_matches = len(matches)

    for idx, row in enumerate(
        matches.itertuples(index=False),
        start=1
    ):

        winner_key = row.winner_key
        loser_key = row.loser_key

        match_key = row.match_key

        # ------------------------------
        # Winner Perspective
        # ------------------------------

        winner_record = tracker.get(
            (winner_key, loser_key),
            {
                "wins": 0,
                "losses": 0
            }
        )

        winner_wins = winner_record["wins"]
        winner_losses = winner_record["losses"]

        winner_matches = (
            winner_wins
            + winner_losses
        )

        winner_pct = (
            winner_wins
            / winner_matches
            * 100
            if winner_matches > 0
            else 50
        )

        rows.append(
            {
                "match_key": match_key,
                "player_key": winner_key,
                "opponent_key": loser_key,
                "grass_h2h_wins_before":
                    winner_wins,
                "grass_h2h_losses_before":
                    winner_losses,
                "grass_h2h_matches_before":
                    winner_matches,
                "grass_h2h_win_pct_before":
                    round(winner_pct, 2)
            }
        )

        # ------------------------------
        # Loser Perspective
        # ------------------------------

        loser_record = tracker.get(
            (loser_key, winner_key),
            {
                "wins": 0,
                "losses": 0
            }
        )

        loser_wins = loser_record["wins"]
        loser_losses = loser_record["losses"]

        loser_matches = (
            loser_wins
            + loser_losses
        )

        loser_pct = (
            loser_wins
            / loser_matches
            * 100
            if loser_matches > 0
            else 50
        )

        rows.append(
            {
                "match_key": match_key,
                "player_key": loser_key,
                "opponent_key": winner_key,
                "grass_h2h_wins_before":
                    loser_wins,
                "grass_h2h_losses_before":
                    loser_losses,
                "grass_h2h_matches_before":
                    loser_matches,
                "grass_h2h_win_pct_before":
                    round(loser_pct, 2)
            }
        )

        # ------------------------------
        # Update History
        # ------------------------------

        winner_record["wins"] += 1
        loser_record["losses"] += 1

        tracker[
            (winner_key, loser_key)
        ] = winner_record

        tracker[
            (loser_key, winner_key)
        ] = loser_record

        if idx % 1000 == 0:
            print(
                f"Processed {idx:,}/{total_matches:,}"
            )

    grass_h2h = pd.DataFrame(rows)

    grass_h2h.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nGrass Head To Head V2")

    print(
        grass_h2h.shape
    )

    print(
        grass_h2h.head()
    )

    return grass_h2h


if __name__ == "__main__":
    build_grass_head_to_head_v2()