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
    / "player_master.parquet"
)


def create_player_master():

    df = pd.read_parquet(INPUT_FILE)

    # ==========================
    # Winner Records
    # ==========================

    winners = df[
        [
            "winner_id",
            "winner_name",
            "winner_hand",
            "winner_ht",
            "winner_ioc",
            "tour",
            "match_year"
        ]
    ].copy()

    winners.columns = [
        "player_id",
        "player_name",
        "hand",
        "height",
        "country",
        "tour",
        "match_year"
    ]

    winners["win"] = 1
    winners["loss"] = 0

    # ==========================
    # Loser Records
    # ==========================

    losers = df[
        [
            "loser_id",
            "loser_name",
            "loser_hand",
            "loser_ht",
            "loser_ioc",
            "tour",
            "match_year"
        ]
    ].copy()

    losers.columns = [
        "player_id",
        "player_name",
        "hand",
        "height",
        "country",
        "tour",
        "match_year"
    ]

    losers["win"] = 0
    losers["loss"] = 1

    # ==========================
    # Combine Player Records
    # ==========================

    players = pd.concat(
        [winners, losers],
        ignore_index=True
    )

    # ==========================
    # Overall Career Stats
    # ==========================

    player_master = (
        players
        .groupby(
            [
                "player_id",
                "player_name",
                "tour"
            ],
            as_index=False
        )
        .agg(
            country=("country", "first"),
            hand=("hand", "first"),
            height=("height", "max"),
            first_match_year=("match_year", "min"),
            last_match_year=("match_year", "max"),
            career_wins=("win", "sum"),
            career_losses=("loss", "sum")
        )
    )

    player_master["career_matches"] = (
        player_master["career_wins"]
        + player_master["career_losses"]
    )

    player_master["career_win_pct"] = (
        player_master["career_wins"]
        / player_master["career_matches"]
        * 100
    ).round(2)

    # ==========================
    # Grass Stats
    # ==========================

    winner_surface = df[
        [
            "winner_id",
            "surface"
        ]
    ].copy()

    winner_surface.columns = [
        "player_id",
        "surface"
    ]

    winner_surface["win"] = 1
    winner_surface["loss"] = 0

    loser_surface = df[
        [
            "loser_id",
            "surface"
        ]
    ].copy()

    loser_surface.columns = [
        "player_id",
        "surface"
    ]

    loser_surface["win"] = 0
    loser_surface["loss"] = 1

    surface_players = pd.concat(
        [
            winner_surface,
            loser_surface
        ],
        ignore_index=True
    )

    grass_stats = (
        surface_players[
            surface_players["surface"]
            .astype(str)
            .str.upper()
            == "GRASS"
        ]
        .groupby("player_id")
        .agg(
            career_grass_wins=("win", "sum"),
            career_grass_losses=("loss", "sum")
        )
        .reset_index()
    )

    grass_stats["career_grass_matches"] = (
        grass_stats["career_grass_wins"]
        +
        grass_stats["career_grass_losses"]
    )

    grass_stats["career_grass_win_pct"] = (
        grass_stats["career_grass_wins"]
        /
        grass_stats["career_grass_matches"]
        * 100
    ).round(2)

    # ==========================
    # Merge Grass Stats
    # ==========================

    player_master = player_master.merge(
        grass_stats,
        on="player_id",
        how="left"
    )

    grass_cols = [
        "career_grass_matches",
        "career_grass_wins",
        "career_grass_losses",
        "career_grass_win_pct"
    ]

    player_master[grass_cols] = (
        player_master[grass_cols]
        .fillna(0)
    )

    player_master["player_key"] = (
    player_master["tour"]
    + "_"
    + player_master["player_id"].astype(str)
)

    # ==========================
    # Sort
    # ==========================

    player_master = player_master.sort_values(
        [
            "tour",
            "career_wins"
        ],
        ascending=[
            True,
            False
        ]
    )

    # ==========================
    # Save
    # ==========================

    player_master.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(f"Players: {len(player_master)}")

    print(
        player_master[
            [
                "player_name",
                "tour",
                "career_wins",
                "career_grass_wins",
                "career_grass_win_pct"
            ]
        ].head(10)
    )

    return player_master


if __name__ == "__main__":
    create_player_master()