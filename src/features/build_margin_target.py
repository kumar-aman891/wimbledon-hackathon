from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_final.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_margin.parquet"
)


INVALID_MARKERS = [
    "RET",
    "W/O",
    "WO",
    "ABD",
    "DEF",
    "UNF",
    "LIVE"
]


def parse_score(score):
    """
    Parse a tennis score and return:

    winner_sets,
    loser_sets

    Returns
    -------
    (None, None)
        if the match is unfinished.
    """

    if pd.isna(score):
        return None, None

    score = str(score).upper().strip()

    for marker in INVALID_MARKERS:
        if marker in score:
            return None, None

    winner_sets = 0
    loser_sets = 0

    sets = score.split()

    for s in sets:

        # remove tiebreak information
        s = re.sub(r"\(.*?\)", "", s)

        if "-" not in s:
            continue

        try:

            left, right = s.split("-")

            left = int(left)
            right = int(right)

        except Exception:

            continue

        if left > right:
            winner_sets += 1

        elif right > left:
            loser_sets += 1

    if winner_sets == 0:
        return None, None

    return winner_sets, loser_sets


def build_margin_target(df):

    winner_sets = []
    loser_sets = []

    for score in df["score"]:

        w, l = parse_score(score)

        winner_sets.append(w)
        loser_sets.append(l)

    df["winner_sets"] = winner_sets
    df["loser_sets"] = loser_sets

    df = (
        df
        .dropna(
            subset=[
                "winner_sets",
                "loser_sets"
            ]
        )
        .copy()
    )

    df["winner_sets"] = (
        df["winner_sets"]
        .astype(int)
    )

    df["loser_sets"] = (
        df["loser_sets"]
        .astype(int)
    )

    #####################################################
    #
    # Binary Margin Target
    #
    # 0 = Exactly 1 Set
    # 1 = Greater than 1 Set
    #
    #####################################################

    df["winning_margin_gt1"] = (

        (
            df["winner_sets"]
            -
            df["loser_sets"]
        ) > 1

    ).astype(int)

    return df


def print_summary(df):

    print()

    print("=" * 60)

    print("Margin Target Distribution")

    print("=" * 60)

    print()

    print(
        df["winning_margin_gt1"]
        .value_counts()
        .sort_index()
    )

    print()

    print("=" * 60)

    print("Men")

    print("=" * 60)

    men = df[
        df["tour"] == "ATP"
    ]

    print(

        men[
            "winning_margin_gt1"
        ].value_counts()

    )

    print()

    print("=" * 60)

    print("Women")

    print("=" * 60)

    women = df[
        df["tour"] == "WTA"
    ]

    print(

        women[
            "winning_margin_gt1"
        ].value_counts()

    )

    print()

    print("=" * 60)

    print("Examples")

    print("=" * 60)

    print(

        df[
            [
                "tour",
                "score",
                "winner_sets",
                "loser_sets",
                "winning_margin_gt1"
            ]
        ]
        .head(20)

    )


def main():

    print()

    print("Loading match dataset...")

    df = pd.read_parquet(
        INPUT_FILE
    )

    print()

    print(
        f"Original Matches : {len(df):,}"
    )

    ####################################################
    #
    # Train only on Grand Slam matches
    #
    ####################################################

    df = df[
        df["is_grand_slam"] == 1
    ].copy()

    print(
        f"Grand Slam Matches : {len(df):,}"
    )

    df = build_margin_target(df)

    print_summary(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()

    print("=" * 60)

    print("Saved")

    print(OUTPUT_FILE)

    print("=" * 60)

    print()


if __name__ == "__main__":

    main()