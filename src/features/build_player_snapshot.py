from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_master.parquet"
)

ELO_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_elo.parquet"
)

FORM_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recent_form.parquet"
)

MATCH_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_final.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_snapshot.parquet"
)


def build_player_snapshot():

    print("Loading datasets...")

    players = pd.read_parquet(
        PLAYER_FILE
    )

    elo = pd.read_parquet(
        ELO_FILE
    )

    recent_form = pd.read_parquet(
        FORM_FILE
    )

    match_features = pd.read_parquet(
        MATCH_FEATURE_FILE
    )

    # ----------------------------------
    # Elo
    # ----------------------------------

    snapshot = players.merge(
        elo,
        on="player_id",
        how="left"
    )

    # ----------------------------------
    # Latest Recent Form
    # ----------------------------------

    recent_form["match_date"] = pd.to_datetime(
        recent_form["match_date"]
    )

    latest_form = (
        recent_form
        .sort_values("match_date")
        .groupby("player_key")
        .tail(1)
    )

    latest_form = latest_form[
        [
            "player_key",
            "win_pct_last_5",
            "win_pct_last_10",
            "grass_win_pct_last_10"
        ]
    ]

    snapshot = snapshot.merge(
        latest_form,
        on="player_key",
        how="left"
    )

    # ----------------------------------
    # Latest Rank + Age
    # ----------------------------------

    winner_side = match_features[
        [
            "winner_id",
            "winner_age",
            "winner_rank",
            "tourney_date"
        ]
    ].rename(
        columns={
            "winner_id": "player_id",
            "winner_age": "age",
            "winner_rank": "rank"
        }
    )

    loser_side = match_features[
        [
            "loser_id",
            "loser_age",
            "loser_rank",
            "tourney_date"
        ]
    ].rename(
        columns={
            "loser_id": "player_id",
            "loser_age": "age",
            "loser_rank": "rank"
        }
    )

    age_rank = pd.concat(
        [
            winner_side,
            loser_side
        ],
        ignore_index=True
    )

    age_rank["tourney_date"] = pd.to_datetime(
        age_rank["tourney_date"]
    )

    latest_age_rank = (
        age_rank
        .sort_values("tourney_date")
        .groupby("player_id")
        .tail(1)
    )

    latest_age_rank = latest_age_rank[
        [
            "player_id",
            "age",
            "rank"
        ]
    ]

    snapshot = snapshot.merge(
        latest_age_rank,
        on="player_id",
        how="left"
    )

    # ----------------------------------
    # Grass Specialization
    # ----------------------------------

    snapshot["grass_specialization"] = (
        snapshot["career_grass_win_pct"]
        -
        snapshot["career_win_pct"]
    )

    # ----------------------------------
    # Fill Missing
    # ----------------------------------

    numeric_cols = snapshot.select_dtypes(
        include="number"
    ).columns

    snapshot[numeric_cols] = (
        snapshot[numeric_cols]
        .fillna(0)
    )

    # ----------------------------------
    # Save
    # ----------------------------------

    snapshot.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nPlayer Snapshot")

    print(
        snapshot.shape
    )

    print("\nColumns")

    print(
        snapshot.columns.tolist()
    )

    print("\nSample")

    print(
        snapshot[
            [
                "player_name",
                "overall_elo",
                "grass_elo",
                "career_win_pct",
                "win_pct_last_10",
                "rank"
            ]
        ]
        .head()
    )

    return snapshot


if __name__ == "__main__":
    build_player_snapshot()