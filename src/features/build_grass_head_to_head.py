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
    / "grass_head_to_head.parquet"
)


def build_grass_head_to_head():

    matches = pd.read_parquet(
        MATCH_FILE
    )

    # ----------------------------------
    # Grass Matches Only
    # ----------------------------------

    matches = matches[
        matches["surface"]
        .str.upper()
        == "GRASS"
    ].copy()

    # ----------------------------------
    # ATP/WTA Safe Keys
    # ----------------------------------

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

    # ----------------------------------
    # Winner Perspective
    # ----------------------------------

    winner_view = matches[
        [
            "winner_key",
            "loser_key"
        ]
    ].copy()

    winner_view.columns = [
        "player_key",
        "opponent_key"
    ]

    winner_view["win"] = 1
    winner_view["loss"] = 0

    # ----------------------------------
    # Loser Perspective
    # ----------------------------------

    loser_view = matches[
        [
            "loser_key",
            "winner_key"
        ]
    ].copy()

    loser_view.columns = [
        "player_key",
        "opponent_key"
    ]

    loser_view["win"] = 0
    loser_view["loss"] = 1

    # ----------------------------------
    # Combine
    # ----------------------------------

    h2h = pd.concat(
        [
            winner_view,
            loser_view
        ],
        ignore_index=True
    )

    # ----------------------------------
    # Aggregate
    # ----------------------------------

    h2h = (
        h2h
        .groupby(
            [
                "player_key",
                "opponent_key"
            ],
            as_index=False
        )
        .agg(
            grass_h2h_wins=("win", "sum"),
            grass_h2h_losses=("loss", "sum")
        )
    )

    h2h["grass_h2h_matches"] = (
        h2h["grass_h2h_wins"]
        +
        h2h["grass_h2h_losses"]
    )

    h2h["grass_h2h_win_pct"] = (
        h2h["grass_h2h_wins"]
        /
        h2h["grass_h2h_matches"]
        * 100
    ).round(2)

    h2h.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nGrass Head To Head Dataset")

    print(
        h2h.shape
    )

    print(
        h2h.head()
    )

    return h2h


if __name__ == "__main__":
    build_grass_head_to_head()