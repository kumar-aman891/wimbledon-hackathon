from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features.parquet"
)

RECENT_FORM_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recent_form.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_v3.parquet"
)


def build_match_features_v3():

    matches = pd.read_parquet(MATCH_FILE)

    recent_form = pd.read_parquet(
        RECENT_FORM_FILE
    )

    matches["match_key"] = (
        matches["tourney_id"].astype(str)
        + "_"
        + matches["match_num"].astype(str)
    )

    # Winner form

    winner_form = (
        recent_form
        .add_prefix("winner_")
    )

    matches = matches.merge(
        winner_form,
        left_on=[
            "match_key",
            "winner_id"
        ],
        right_on=[
            "winner_match_key",
            "winner_player_id"
        ],
        how="left"
    )

    # Loser form

    loser_form = (
        recent_form
        .add_prefix("loser_")
    )

    matches = matches.merge(
        loser_form,
        left_on=[
            "match_key",
            "loser_id"
        ],
        right_on=[
            "loser_match_key",
            "loser_player_id"
        ],
        how="left"
    )

    # Derived form features

    matches["form_diff_5"] = (
        matches["winner_win_pct_last_5"]
        -
        matches["loser_win_pct_last_5"]
    )

    matches["form_diff_10"] = (
        matches["winner_win_pct_last_10"]
        -
        matches["loser_win_pct_last_10"]
    )

    matches["grass_form_diff_10"] = (
        matches["winner_grass_win_pct_last_10"]
        -
        matches["loser_grass_win_pct_last_10"]
    )

    matches.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature Store V3")
    print(matches.shape)

    return matches


if __name__ == "__main__":
    build_match_features_v3()