from pathlib import Path
import re

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SNAPSHOT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_snapshot.parquet"
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

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "catboost_final.pkl"
)

MEN_PARTICIPANTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "men_participants.csv"
)

WOMEN_PARTICIPANTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "women_participants.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MEN_OUTPUT_FILE = (
    OUTPUT_DIR
    / "men_contender_rankings.csv"
)

WOMEN_OUTPUT_FILE = (
    OUTPUT_DIR
    / "women_contender_rankings.csv"
)

TOP16_OUTPUT_FILE = (
    OUTPUT_DIR
    / "top_16_contenders.csv"
)


ENTRY_SUFFIXES = {
    "SR",
    "PR",
    "WC",
    "Q",
    "LL",
    "ALT",
    "SE",
    "JR"
}


PARTICIPANT_NAME_ALIASES = {
    # women
    "xinyuwang": "Xinyu Wang",
    "elenagabrielaruse": "Elena-Gabriela Ruse",
    "irinacameliabegu": "Irina-Camelia Begu",
    "sarasorribestormo": "Sara Sorribes Tormo",
    "nadiapodoroska": "Nadia Podoroska",
    "aoiito": "Aoi Ito",
    "oleksandraoliynykova": "Oleksandra Oliynykova",
    "mimixu": "Mimi Xu",
    "aliciadudeney": "Alicia Dudeney",

    # men
    "tomasmartinetcheverry": "Tomas Martin Etcheverry",
    "camilougocarabelli": "Camilo Ugo Carabelli",
    "robertobautistaagut": "Roberto Bautista Agut",
    "alejandrodavidovichfokina": "Alejandro Davidovich Fokina",
    "felixaugeraliassime": "Felix Auger-Aliassime",
    "jackpinningtonjones": "Jack Pinnington Jones",
    "giovannimpetshiperricard": "Giovanni Mpetshi Perricard",
    "aleksandrshevchenko": "Alexander Shevchenko"
}


FALLBACK_NUMERIC_COLUMNS = [
    "overall_elo",
    "grass_elo",
    "career_wins",
    "career_losses",
    "career_matches",
    "career_win_pct",
    "career_grass_wins",
    "career_grass_losses",
    "career_grass_matches",
    "career_grass_win_pct",
    "win_pct_last_5",
    "win_pct_last_10",
    "grass_win_pct_last_10",
    "age",
    "rank",
    "grass_specialization"
]


def strip_entry_suffixes(name):

    if pd.isna(name):
        return None

    name = str(name).strip()

    while True:

        parts = name.split()

        if len(parts) == 0:
            break

        last_token = parts[-1].upper()

        if last_token in ENTRY_SUFFIXES:
            parts = parts[:-1]
            name = " ".join(parts).strip()
        else:
            break

    return name


def clean_participant_display_name(name):

    if pd.isna(name):
        return None

    name = str(name).strip()

    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    name = strip_entry_suffixes(name)
    name = re.sub(r"\s+", " ", name).strip()

    if "," in name:

        last_name, first_name = name.split(",", 1)

        last_name = last_name.strip().title()
        first_name = first_name.strip().title()

        name = f"{first_name} {last_name}"

    else:
        name = name.title()

    name = re.sub(r"\s+", " ", name).strip()

    return name


def normalize_snapshot_name(name):

    if pd.isna(name):
        return None

    name = str(name).strip()
    name = re.sub(r"\s+", " ", name).strip()

    return name


def build_match_key(name):

    if pd.isna(name):
        return None

    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]", "", name)

    return name


def snapshot_match_key(name):

    if pd.isna(name):
        return None

    name = normalize_snapshot_name(name)
    return build_match_key(name)


def get_h2h_feature(
    h2h_df,
    player_key,
    opponent_key,
    column_name
):

    row = h2h_df[
        (h2h_df["player_key"] == player_key)
        &
        (h2h_df["opponent_key"] == opponent_key)
    ]

    if len(row) == 0:
        return 50.0

    row = (
        row
        .sort_values("match_key")
        .tail(1)
    )

    return float(
        row.iloc[0][column_name]
    )


def build_feature_vector(
    p1,
    p2,
    h2h_df,
    grass_h2h_df,
    surface="GRASS",
    round_name="R128",
    tournament="GRAND_SLAM"
):

    features = {}

    features["elo_diff"] = (
        p1["overall_elo"]
        -
        p2["overall_elo"]
    )

    features["grass_elo_diff"] = (
        p1["grass_elo"]
        -
        p2["grass_elo"]
    )

    features["rank_diff"] = (
        p2["rank"]
        -
        p1["rank"]
    )

    features["career_win_pct_diff"] = (
        p1["career_win_pct"]
        -
        p2["career_win_pct"]
    )

    features["grass_win_pct_diff"] = (
        p1["career_grass_win_pct"]
        -
        p2["career_grass_win_pct"]
    )

    features["experience_diff"] = (
        p1["career_matches"]
        -
        p2["career_matches"]
    )

    features["form_diff_5"] = (
        p1["win_pct_last_5"]
        -
        p2["win_pct_last_5"]
    )

    features["form_diff_10"] = (
        p1["win_pct_last_10"]
        -
        p2["win_pct_last_10"]
    )

    features["grass_form_diff_10"] = (
        p1["grass_win_pct_last_10"]
        -
        p2["grass_win_pct_last_10"]
    )

    features["age_diff"] = (
        p1["age"]
        -
        p2["age"]
    )

    features["grass_specialization_diff"] = (
        p1["grass_specialization"]
        -
        p2["grass_specialization"]
    )

    p1_h2h = get_h2h_feature(
        h2h_df,
        p1["player_key"],
        p2["player_key"],
        "h2h_win_pct_before"
    )

    p2_h2h = get_h2h_feature(
        h2h_df,
        p2["player_key"],
        p1["player_key"],
        "h2h_win_pct_before"
    )

    features["h2h_win_pct_diff"] = (
        p1_h2h
        -
        p2_h2h
    )

    p1_grass_h2h = get_h2h_feature(
        grass_h2h_df,
        p1["player_key"],
        p2["player_key"],
        "grass_h2h_win_pct_before"
    )

    p2_grass_h2h = get_h2h_feature(
        grass_h2h_df,
        p2["player_key"],
        p1["player_key"],
        "grass_h2h_win_pct_before"
    )

    features["grass_h2h_win_pct_diff"] = (
        p1_grass_h2h
        -
        p2_grass_h2h
    )

    features["is_grand_slam"] = int(
        tournament == "GRAND_SLAM"
    )

    features["is_masters"] = int(
        tournament == "MASTERS"
    )

    features["is_atp500"] = int(
        tournament == "ATP500"
    )

    features["is_atp250"] = int(
        tournament == "ATP250"
    )

    round_cols = [
        "round_BR",
        "round_F",
        "round_QF",
        "round_R128",
        "round_R16",
        "round_R32",
        "round_R64",
        "round_RR",
        "round_SF"
    ]

    for col in round_cols:
        features[col] = 0

    if f"round_{round_name}" in round_cols:
        features[f"round_{round_name}"] = 1

    surface_cols = [
        "surface_Carpet",
        "surface_Clay",
        "surface_Grass",
        "surface_Hard",
        "surface_clay"
    ]

    for col in surface_cols:
        features[col] = 0

    if surface == "GRASS":
        features["surface_Grass"] = 1

    elif surface == "CLAY":
        features["surface_Clay"] = 1

    elif surface == "HARD":
        features["surface_Hard"] = 1

    return pd.DataFrame([features])


def create_fallback_profile(
    player_name,
    tour,
    tour_snapshot,
    participant_match_key
):

    medians = {}

    for col in FALLBACK_NUMERIC_COLUMNS:

        if col in tour_snapshot.columns:
            medians[col] = pd.to_numeric(
                tour_snapshot[col],
                errors="coerce"
            ).median()
        else:
            medians[col] = 0.0

    fallback = {
        "player_id": f"FALLBACK_{tour}_{participant_match_key}",
        "player_name": player_name,
        "tour": tour,
        "country": "UNK",
        "hand": None,
        "height": None,
        "first_match_year": None,
        "last_match_year": None,
        "career_wins": medians["career_wins"],
        "career_losses": medians["career_losses"],
        "career_matches": medians["career_matches"],
        "career_win_pct": medians["career_win_pct"],
        "career_grass_wins": medians["career_grass_wins"],
        "career_grass_losses": medians["career_grass_losses"],
        "career_grass_matches": medians["career_grass_matches"],
        "career_grass_win_pct": medians["career_grass_win_pct"],
        "player_key": f"FALLBACK_{tour}_{participant_match_key}",
        "overall_elo": medians["overall_elo"],
        "grass_elo": medians["grass_elo"],
        "win_pct_last_5": medians["win_pct_last_5"],
        "win_pct_last_10": medians["win_pct_last_10"],
        "grass_win_pct_last_10": medians["grass_win_pct_last_10"],
        "age": medians["age"],
        "rank": medians["rank"],
        "grass_specialization": medians["grass_specialization"]
    }

    return fallback


def preprocess_participants(
    participants_file,
    label,
    truncate_after_raw_name=None
):

    participants = pd.read_csv(participants_file)

    if "Player_name" not in participants.columns:
        raise ValueError(
            f"{participants_file.name} must contain a 'Player_name' column."
        )

    participants = participants.copy()

    participants["raw_player_name"] = participants["Player_name"].astype(str).str.strip()

    if truncate_after_raw_name is not None:

        marker_clean = str(truncate_after_raw_name).strip()

        marker_idx = participants[
            participants["raw_player_name"].str.strip() == marker_clean
        ].index

        if len(marker_idx) == 0:
            print(
                f"Warning: truncate marker '{truncate_after_raw_name}' not found in {label} file. Using full file."
            )
        else:
            last_idx = marker_idx[0]
            participants = participants.loc[:last_idx].copy()

    participants["clean_player_name"] = participants["raw_player_name"].apply(
        clean_participant_display_name
    )

    participants["participant_match_key"] = participants["clean_player_name"].apply(
        build_match_key
    )

    participants = participants[
        participants["participant_match_key"].notna()
    ].copy()

    before = len(participants)

    participants = participants.drop_duplicates(
        subset=["participant_match_key"],
        keep="first"
    ).reset_index(drop=True)

    after = len(participants)

    print(f"\n{label}: participant rows before dedupe = {before}")
    print(f"{label}: participant rows after dedupe  = {after}")

    return participants


def build_field_from_participants(
    participants_file,
    snapshot,
    expected_tour,
    label,
    truncate_after_raw_name=None
):

    participants = preprocess_participants(
        participants_file=participants_file,
        label=label,
        truncate_after_raw_name=truncate_after_raw_name
    )

    participants["alias_snapshot_name"] = participants["participant_match_key"].map(
        PARTICIPANT_NAME_ALIASES
    )

    snapshot_tour = snapshot[
        snapshot["tour"].astype(str).str.upper() == expected_tour.upper()
    ].copy()

    snapshot_tour["snapshot_clean_name"] = snapshot_tour["player_name"].apply(
        normalize_snapshot_name
    )

    snapshot_tour["snapshot_match_key"] = snapshot_tour["player_name"].apply(
        snapshot_match_key
    )

    matched = participants.merge(
        snapshot_tour,
        left_on="participant_match_key",
        right_on="snapshot_match_key",
        how="left",
        suffixes=("_participant", "")
    )

    unmatched_mask = (
        matched["player_name"].isna()
        &
        matched["alias_snapshot_name"].notna()
    )

    if unmatched_mask.any():

        alias_rows = matched.loc[
            unmatched_mask,
            [
                "raw_player_name",
                "clean_player_name",
                "participant_match_key",
                "alias_snapshot_name"
            ]
        ].copy()

        alias_rows = alias_rows.merge(
            snapshot_tour,
            left_on="alias_snapshot_name",
            right_on="player_name",
            how="left"
        )

        for col in snapshot_tour.columns:
            matched.loc[unmatched_mask, col] = alias_rows[col].values

    unmatched = matched[
        matched["player_name"].isna()
    ][[
        "raw_player_name",
        "clean_player_name",
        "participant_match_key"
    ]].drop_duplicates()

    if len(unmatched) > 0:
        print(f"\nUnmatched {label} participants before fallback:")
        print(unmatched.to_string(index=False))

    field = matched[
        matched["player_name"].notna()
    ].copy()

    field = field.drop_duplicates(
        subset=["player_key"]
    ).reset_index(drop=True)

    fallback_rows = []

    if len(unmatched) > 0:

        for _, row in unmatched.iterrows():

            fallback_rows.append(
                create_fallback_profile(
                    player_name=row["clean_player_name"],
                    tour=expected_tour.upper(),
                    tour_snapshot=snapshot_tour,
                    participant_match_key=row["participant_match_key"]
                )
            )

    if len(fallback_rows) > 0:

        fallback_df = pd.DataFrame(fallback_rows)

        field = pd.concat(
            [
                field,
                fallback_df
            ],
            ignore_index=True
        )

        print(f"{label}: added {len(fallback_df)} fallback player profiles")

    field = field.drop_duplicates(
        subset=["player_key"]
    ).reset_index(drop=True)

    print(f"\n{label} final field size: {len(field)} players")

    return field, unmatched


def score_field(
    field_df,
    model,
    h2h_df,
    grass_h2h_df,
    label
):

    rows = []

    total_players = len(field_df)

    for i, (_, p1) in enumerate(field_df.iterrows(), start=1):

        probs = []

        for _, p2 in field_df.iterrows():

            if p1["player_key"] == p2["player_key"]:
                continue

            X = build_feature_vector(
                p1=p1,
                p2=p2,
                h2h_df=h2h_df,
                grass_h2h_df=grass_h2h_df,
                surface="GRASS",
                round_name="R128",
                tournament="GRAND_SLAM"
            )

            prob = float(
                model.predict_proba(X)[0][1]
            )

            probs.append(prob)

        avg_prob = float(pd.Series(probs).mean()) if probs else None
        median_prob = float(pd.Series(probs).median()) if probs else None
        p25_prob = float(pd.Series(probs).quantile(0.25)) if probs else None
        p75_prob = float(pd.Series(probs).quantile(0.75)) if probs else None

        rows.append(
            {
                "tour_group": label,
                "player_name": p1["player_name"],
                "tour": p1["tour"],
                "rank": p1["rank"],
                "overall_elo": p1["overall_elo"],
                "grass_elo": p1["grass_elo"],
                "career_win_pct": p1["career_win_pct"],
                "career_grass_win_pct": p1["career_grass_win_pct"],
                "win_pct_last_10": p1["win_pct_last_10"],
                "grass_win_pct_last_10": p1["grass_win_pct_last_10"],
                "grass_specialization": p1["grass_specialization"],
                "field_avg_win_prob": avg_prob,
                "field_median_win_prob": median_prob,
                "field_p25_win_prob": p25_prob,
                "field_p75_win_prob": p75_prob,
                "is_fallback_profile": int(
                    str(p1["player_key"]).startswith("FALLBACK_")
                )
            }
        )

        print(
            f"{label}: processed {i}/{total_players} - {p1['player_name']}"
        )

    ranking = pd.DataFrame(rows)

    ranking = ranking.sort_values(
        [
            "field_avg_win_prob",
            "field_median_win_prob",
            "grass_elo",
            "overall_elo"
        ],
        ascending=False
    ).reset_index(drop=True)

    ranking["field_rank"] = ranking.index + 1

    return ranking


def main():

    print("Loading model and data...")

    snapshot = pd.read_parquet(SNAPSHOT_FILE)
    h2h_df = pd.read_parquet(H2H_FILE)
    grass_h2h_df = pd.read_parquet(GRASS_H2H_FILE)
    model = joblib.load(MODEL_FILE)

    men_field, men_unmatched = build_field_from_participants(
        participants_file=MEN_PARTICIPANTS_FILE,
        snapshot=snapshot,
        expected_tour="ATP",
        label="MEN",
        truncate_after_raw_name=None
    )

    women_field, women_unmatched = build_field_from_participants(
        participants_file=WOMEN_PARTICIPANTS_FILE,
        snapshot=snapshot,
        expected_tour="WTA",
        label="WOMEN",
        truncate_after_raw_name="WILLIAMS, Serena (USA)"
    )

    print("\nScoring men's field...")
    men_rankings = score_field(
        field_df=men_field,
        model=model,
        h2h_df=h2h_df,
        grass_h2h_df=grass_h2h_df,
        label="MEN"
    )

    print("\nScoring women's field...")
    women_rankings = score_field(
        field_df=women_field,
        model=model,
        h2h_df=h2h_df,
        grass_h2h_df=grass_h2h_df,
        label="WOMEN"
    )

    men_rankings.to_csv(
        MEN_OUTPUT_FILE,
        index=False
    )

    women_rankings.to_csv(
        WOMEN_OUTPUT_FILE,
        index=False
    )

    top_8_men = men_rankings.head(8).copy()
    top_8_men["selection_group"] = "Top 8 Men"

    top_8_women = women_rankings.head(8).copy()
    top_8_women["selection_group"] = "Top 8 Women"

    top_16 = pd.concat(
        [
            top_8_men,
            top_8_women
        ],
        ignore_index=True
    )

    top_16.to_csv(
        TOP16_OUTPUT_FILE,
        index=False
    )

    print("\nTop 8 Men")
    print(
        men_rankings[
            [
                "field_rank",
                "player_name",
                "field_avg_win_prob",
                "grass_elo",
                "rank",
                "is_fallback_profile"
            ]
        ].head(8).to_string(index=False)
    )

    print("\nTop 8 Women")
    print(
        women_rankings[
            [
                "field_rank",
                "player_name",
                "field_avg_win_prob",
                "grass_elo",
                "rank",
                "is_fallback_profile"
            ]
        ].head(8).to_string(index=False)
    )

    print("\nFiles written:")
    print(MEN_OUTPUT_FILE)
    print(WOMEN_OUTPUT_FILE)
    print(TOP16_OUTPUT_FILE)

    if len(men_unmatched) > 0 or len(women_unmatched) > 0:
        print("\nNote: unmatched players were included using fallback median profiles.")


if __name__ == "__main__":
    main()