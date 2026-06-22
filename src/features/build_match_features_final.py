from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_v5.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_final.parquet"
)


def build_match_features_final():

    print("Loading feature store...")

    df = pd.read_parquet(INPUT_FILE)

    # --------------------------------------------------
    # AGE FEATURES
    # --------------------------------------------------

    if (
        "winner_age" in df.columns
        and
        "loser_age" in df.columns
    ):

        df["age_diff"] = (
            df["winner_age"]
            -
            df["loser_age"]
        )

    # --------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------

    if (
        "winner_career_matches" in df.columns
        and
        "loser_career_matches" in df.columns
    ):

        df["experience_diff"] = (
            df["winner_career_matches"]
            -
            df["loser_career_matches"]
        )

    # --------------------------------------------------
    # GRASS SPECIALIZATION
    # --------------------------------------------------

    if (
        "winner_career_grass_win_pct" in df.columns
        and
        "winner_career_win_pct" in df.columns
    ):

        df["winner_grass_specialization"] = (
            df["winner_career_grass_win_pct"]
            -
            df["winner_career_win_pct"]
        )

        df["loser_grass_specialization"] = (
            df["loser_career_grass_win_pct"]
            -
            df["loser_career_win_pct"]
        )

        df["grass_specialization_diff"] = (
            df["winner_grass_specialization"]
            -
            df["loser_grass_specialization"]
        )

    # --------------------------------------------------
    # TOURNAMENT LEVEL
    # --------------------------------------------------

    if "tourney_level" in df.columns:

        df["is_grand_slam"] = (
            df["tourney_level"]
            == "G"
        ).astype(int)

        df["is_masters"] = (
            df["tourney_level"]
            == "M"
        ).astype(int)

        df["is_atp500"] = (
            df["tourney_level"]
            == "A"
        ).astype(int)

        df["is_atp250"] = (
            df["tourney_level"]
            == "B"
        ).astype(int)

    # --------------------------------------------------
    # ROUND DUMMIES
    # --------------------------------------------------

    if "round" in df.columns:

        round_dummies = pd.get_dummies(
            df["round"],
            prefix="round",
            dtype=int
        )

        df = pd.concat(
            [
                df,
                round_dummies
            ],
            axis=1
        )

    # --------------------------------------------------
    # SURFACE DUMMIES
    # --------------------------------------------------

    if "surface" in df.columns:

        surface_dummies = pd.get_dummies(
            df["surface"],
            prefix="surface",
            dtype=int
        )

        df = pd.concat(
            [
                df,
                surface_dummies
            ],
            axis=1
        )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nFinal Feature Store")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


if __name__ == "__main__":
    build_match_features_final()