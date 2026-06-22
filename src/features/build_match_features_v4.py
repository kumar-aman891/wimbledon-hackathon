from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_v3.parquet"
)

H2H_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "head_to_head.parquet"
)

GRASS_H2H_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "grass_head_to_head.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_v4.parquet"
)


def build_match_features_v4():

    print("Loading datasets...")

    matches = pd.read_parquet(
        MATCH_FILE
    )

    h2h = pd.read_parquet(
        H2H_FILE
    )

    grass_h2h = pd.read_parquet(
        GRASS_H2H_FILE
    )

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
    # Winner H2H
    # ----------------------------------

    winner_h2h = h2h.rename(
        columns={
            "player_key": "winner_key",
            "opponent_key": "loser_key",
            "h2h_win_pct": "winner_h2h_win_pct"
        }
    )[
        [
            "winner_key",
            "loser_key",
            "winner_h2h_win_pct"
        ]
    ]

    matches = matches.merge(
        winner_h2h,
        on=[
            "winner_key",
            "loser_key"
        ],
        how="left"
    )

    # ----------------------------------
    # Loser H2H
    # ----------------------------------

    loser_h2h = h2h.rename(
        columns={
            "player_key": "loser_key",
            "opponent_key": "winner_key",
            "h2h_win_pct": "loser_h2h_win_pct"
        }
    )[
        [
            "winner_key",
            "loser_key",
            "loser_h2h_win_pct"
        ]
    ]

    matches = matches.merge(
        loser_h2h,
        on=[
            "winner_key",
            "loser_key"
        ],
        how="left"
    )

    # ----------------------------------
    # Winner Grass H2H
    # ----------------------------------

    winner_grass_h2h = grass_h2h.rename(
        columns={
            "player_key": "winner_key",
            "opponent_key": "loser_key",
            "grass_h2h_win_pct":
                "winner_grass_h2h_win_pct"
        }
    )[
        [
            "winner_key",
            "loser_key",
            "winner_grass_h2h_win_pct"
        ]
    ]

    matches = matches.merge(
        winner_grass_h2h,
        on=[
            "winner_key",
            "loser_key"
        ],
        how="left"
    )

    # ----------------------------------
    # Loser Grass H2H
    # ----------------------------------

    loser_grass_h2h = grass_h2h.rename(
        columns={
            "player_key": "loser_key",
            "opponent_key": "winner_key",
            "grass_h2h_win_pct":
                "loser_grass_h2h_win_pct"
        }
    )[
        [
            "winner_key",
            "loser_key",
            "loser_grass_h2h_win_pct"
        ]
    ]

    matches = matches.merge(
        loser_grass_h2h,
        on=[
            "winner_key",
            "loser_key"
        ],
        how="left"
    )

    # ----------------------------------
    # Missing H2H = Neutral
    # ----------------------------------

    matches[
        [
            "winner_h2h_win_pct",
            "loser_h2h_win_pct",
            "winner_grass_h2h_win_pct",
            "loser_grass_h2h_win_pct"
        ]
    ] = matches[
        [
            "winner_h2h_win_pct",
            "loser_h2h_win_pct",
            "winner_grass_h2h_win_pct",
            "loser_grass_h2h_win_pct"
        ]
    ].fillna(50)

    # ----------------------------------
    # New Features
    # ----------------------------------

    matches["h2h_win_pct_diff"] = (
        matches["winner_h2h_win_pct"]
        -
        matches["loser_h2h_win_pct"]
    )

    matches["grass_h2h_win_pct_diff"] = (
        matches["winner_grass_h2h_win_pct"]
        -
        matches["loser_grass_h2h_win_pct"]
    )

    matches.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature Store V4")
    print(matches.shape)

    print("\nNew Features")
    print(
        [
            "h2h_win_pct_diff",
            "grass_h2h_win_pct_diff"
        ]
    )

    return matches


if __name__ == "__main__":
    build_match_features_v4()