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
    / "head_to_head_v2.parquet"
)

GRASS_H2H_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "grass_head_to_head_v2.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_v5.parquet"
)


def build_match_features_v5():

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

    # --------------------------------------------------
    # Keys
    # --------------------------------------------------

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

    # --------------------------------------------------
    # H2H Winner
    # --------------------------------------------------

    winner_h2h = (
        h2h[
            [
                "match_key",
                "player_key",
                "h2h_win_pct_before"
            ]
        ]
        .rename(
            columns={
                "player_key":
                    "winner_key",
                "h2h_win_pct_before":
                    "winner_h2h_win_pct_before"
            }
        )
    )

    matches = matches.merge(
        winner_h2h,
        on=[
            "match_key",
            "winner_key"
        ],
        how="left"
    )

    # --------------------------------------------------
    # H2H Loser
    # --------------------------------------------------

    loser_h2h = (
        h2h[
            [
                "match_key",
                "player_key",
                "h2h_win_pct_before"
            ]
        ]
        .rename(
            columns={
                "player_key":
                    "loser_key",
                "h2h_win_pct_before":
                    "loser_h2h_win_pct_before"
            }
        )
    )

    matches = matches.merge(
        loser_h2h,
        on=[
            "match_key",
            "loser_key"
        ],
        how="left"
    )

    # --------------------------------------------------
    # Grass H2H Winner
    # --------------------------------------------------

    winner_grass_h2h = (
        grass_h2h[
            [
                "match_key",
                "player_key",
                "grass_h2h_win_pct_before"
            ]
        ]
        .rename(
            columns={
                "player_key":
                    "winner_key",
                "grass_h2h_win_pct_before":
                    "winner_grass_h2h_win_pct_before"
            }
        )
    )

    matches = matches.merge(
        winner_grass_h2h,
        on=[
            "match_key",
            "winner_key"
        ],
        how="left"
    )

    # --------------------------------------------------
    # Grass H2H Loser
    # --------------------------------------------------

    loser_grass_h2h = (
        grass_h2h[
            [
                "match_key",
                "player_key",
                "grass_h2h_win_pct_before"
            ]
        ]
        .rename(
            columns={
                "player_key":
                    "loser_key",
                "grass_h2h_win_pct_before":
                    "loser_grass_h2h_win_pct_before"
            }
        )
    )

    matches = matches.merge(
        loser_grass_h2h,
        on=[
            "match_key",
            "loser_key"
        ],
        how="left"
    )

    # --------------------------------------------------
    # Missing = Neutral
    # --------------------------------------------------

    fill_cols = [
        "winner_h2h_win_pct_before",
        "loser_h2h_win_pct_before",
        "winner_grass_h2h_win_pct_before",
        "loser_grass_h2h_win_pct_before"
    ]

    matches[fill_cols] = (
        matches[fill_cols]
        .fillna(50)
    )

    # --------------------------------------------------
    # Derived Features
    # --------------------------------------------------

    matches["h2h_win_pct_diff"] = (
        matches["winner_h2h_win_pct_before"]
        -
        matches["loser_h2h_win_pct_before"]
    )

    matches["grass_h2h_win_pct_diff"] = (
        matches["winner_grass_h2h_win_pct_before"]
        -
        matches["loser_grass_h2h_win_pct_before"]
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    matches.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature Store V5")

    print(
        f"Rows: {len(matches):,}"
    )

    print(
        f"Columns: {len(matches.columns)}"
    )

    print("\nNull Check")

    print(
        matches[
            [
                "winner_h2h_win_pct_before",
                "loser_h2h_win_pct_before",
                "winner_grass_h2h_win_pct_before",
                "loser_grass_h2h_win_pct_before"
            ]
        ]
        .isna()
        .sum()
    )

    print("\nFeature Sample")

    print(
        matches[
            [
                "h2h_win_pct_diff",
                "grass_h2h_win_pct_diff"
            ]
        ]
        .head()
    )

    return matches


if __name__ == "__main__":
    build_match_features_v5()