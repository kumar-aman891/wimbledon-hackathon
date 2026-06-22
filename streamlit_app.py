from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

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


# ==========================================================
# Cached Loads
# ==========================================================

@st.cache_data
def load_snapshot():
    return pd.read_parquet(SNAPSHOT_FILE)


@st.cache_data
def load_h2h():
    return pd.read_parquet(H2H_FILE)


@st.cache_data
def load_grass_h2h():
    return pd.read_parquet(GRASS_H2H_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


snapshot = load_snapshot()
h2h_df = load_h2h()
grass_h2h_df = load_grass_h2h()
model = load_model()


# ==========================================================
# UI Setup
# ==========================================================

st.set_page_config(
    page_title="Wimbledon AI Match Predictor",
    layout="wide"
)

st.title("🏆 Wimbledon AI Match Predictor")

st.caption(
    "ATP + WTA | Elo Ratings | Recent Form | Head-to-Head | CatBoost Machine Learning"
)


# ==========================================================
# Player List
# ==========================================================

players = (
    snapshot
    .sort_values(
        "overall_elo",
        ascending=False
    )
    ["player_name"]
    .dropna()
    .unique()
    .tolist()
)


# ==========================================================
# Input Controls
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    player1_name = st.selectbox(
        "Player 1",
        players,
        index=0
    )

player2_options = [
    p
    for p in players
    if p != player1_name
]

with col2:

    player2_name = st.selectbox(
        "Player 2",
        player2_options,
        index=0
    )

surface = st.selectbox(
    "Surface",
    [
        "GRASS",
        "CLAY",
        "HARD"
    ]
)

round_name = st.selectbox(
    "Round",
    [
        "R128",
        "R64",
        "R32",
        "R16",
        "QF",
        "SF",
        "F"
    ]
)

tournament = st.selectbox(
    "Tournament",
    [
        "GRAND_SLAM",
        "MASTERS",
        "ATP500",
        "ATP250"
    ]
)


# ==========================================================
# Player Profiles
# ==========================================================

p1_profile = snapshot[
    snapshot["player_name"] == player1_name
].iloc[0]

p2_profile = snapshot[
    snapshot["player_name"] == player2_name
].iloc[0]

st.subheader("Player Profiles")

profile_col1, profile_col2 = st.columns(2)

with profile_col1:

    st.markdown(
        f"""
### {p1_profile['player_name']}

**Rank:** {int(p1_profile['rank'])}

**Overall Elo:** {p1_profile['overall_elo']:.0f}

**Grass Elo:** {p1_profile['grass_elo']:.0f}

**Career Win %:** {p1_profile['career_win_pct']:.1f}

**Recent Form (Last 10):** {p1_profile['win_pct_last_10']:.1f}
"""
    )

with profile_col2:

    st.markdown(
        f"""
### {p2_profile['player_name']}

**Rank:** {int(p2_profile['rank'])}

**Overall Elo:** {p2_profile['overall_elo']:.0f}

**Grass Elo:** {p2_profile['grass_elo']:.0f}

**Career Win %:** {p2_profile['career_win_pct']:.1f}

**Recent Form (Last 10):** {p2_profile['win_pct_last_10']:.1f}
"""
    )


# ==========================================================
# Utility Functions
# ==========================================================

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


# ==========================================================
# Prediction
# ==========================================================

if st.button("Predict Match"):

    p1 = snapshot[
        snapshot["player_name"] == player1_name
    ].iloc[0]

    p2 = snapshot[
        snapshot["player_name"] == player2_name
    ].iloc[0]

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

    X = pd.DataFrame([features])

    probability = (
        model.predict_proba(X)[0][1]
    )

    st.subheader("Win Probability")

    pred_col1, pred_col2 = st.columns(2)

    with pred_col1:
        st.metric(
            player1_name,
            f"{probability:.1%}"
        )

    with pred_col2:
        st.metric(
            player2_name,
            f"{(1 - probability):.1%}"
        )

    st.progress(float(probability))

    st.subheader("Why This Prediction?")

    feature_table = pd.DataFrame(
        {
            "Factor": [
                "Career Win %",
                "Grass Win %",
                "Overall Elo",
                "Grass Elo",
                "Recent Form",
                "Head To Head"
            ],
            "Advantage": [
                X.iloc[0]["career_win_pct_diff"],
                X.iloc[0]["grass_win_pct_diff"],
                X.iloc[0]["elo_diff"],
                X.iloc[0]["grass_elo_diff"],
                X.iloc[0]["form_diff_10"],
                X.iloc[0]["h2h_win_pct_diff"]
            ]
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True
    )

    st.subheader("Model Insights")

    st.markdown(
        """
### Model Summary

**Algorithm:** CatBoost

**ROC-AUC:** 0.7543

### Top Factors Learned

1. Career Win Percentage
2. Player Ranking
3. Elo Rating
4. Grass Court Ability
5. Recent Form
6. Head-to-Head

### Dataset

- ATP + WTA Matches
- 59,312 Historical Matches
- 2,560 Players
- Elo Ratings
- Grass Elo
- Recent Form
- Temporal Head-to-Head
"""
    )