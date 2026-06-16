from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

elo = pd.read_parquet(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_elo.parquet"
)

players = pd.read_parquet(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_master.parquet"
)

result = elo.merge(
    players[
        [
            "player_id",
            "player_name",
            "tour"
        ]
    ],
    on="player_id",
    how="left"
)

print("\nTop Overall Elo\n")

print(
    result.sort_values(
        "overall_elo",
        ascending=False
    )
    [
        [
            "player_name",
            "tour",
            "overall_elo",
            "grass_elo"
        ]
    ]
    .head(30)
)

print("\nTop Grass Elo\n")

print(
    result.sort_values(
        "grass_elo",
        ascending=False
    )
    [
        [
            "player_name",
            "tour",
            "overall_elo",
            "grass_elo"
        ]
    ]
    .head(30)
)