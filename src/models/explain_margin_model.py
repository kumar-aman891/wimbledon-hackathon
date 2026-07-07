from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_margin.parquet"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "catboost_margin.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)


FEATURES = [

    "elo_diff",
    "grass_elo_diff",
    "rank_diff",
    "career_win_pct_diff",
    "grass_win_pct_diff",
    "experience_diff",
    "form_diff_5",
    "form_diff_10",
    "grass_form_diff_10",
    "h2h_win_pct_diff",
    "grass_h2h_win_pct_diff",
    "age_diff",
    "grass_specialization_diff",

    "is_grand_slam",
    "is_masters",
    "is_atp500",
    "is_atp250",

    "round_BR",
    "round_F",
    "round_QF",
    "round_R128",
    "round_R16",
    "round_R32",
    "round_R64",
    "round_RR",
    "round_SF",

    "surface_Carpet",
    "surface_Clay",
    "surface_Grass",
    "surface_Hard",
    "surface_clay"

]


def main():

    print()
    print("Loading model...")

    model = joblib.load(MODEL_FILE)

    print("Loading data...")

    df = pd.read_parquet(DATA_FILE)

    X = df[FEATURES]

    # Keep SHAP runtime reasonable
    SAMPLE_SIZE = min(3000, len(X))
    X_sample = X.sample(
        SAMPLE_SIZE,
        random_state=42
    )

    print()
    print(f"Computing SHAP values on {len(X_sample)} samples...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    ############################################################
    # Feature Importance
    ############################################################

    importance = pd.DataFrame({

        "feature": FEATURES,

        "importance": model.get_feature_importance()

    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 60)
    print("Top Features")
    print("=" * 60)
    print()

    print(importance)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    importance.to_csv(

        OUTPUT_DIR
        / "margin_feature_importance.csv",

        index=False

    )

    ############################################################
    # SHAP Summary Plot
    ############################################################

    plt.figure(figsize=(12, 8))

    shap.summary_plot(

        shap_values,

        X_sample,

        show=False

    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR
        / "margin_shap_summary.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    ############################################################
    # Mean Absolute SHAP
    ############################################################

    mean_abs_shap = (
        abs(shap_values)
        .mean(axis=0)
    )

    shap_df = pd.DataFrame({

        "feature": FEATURES,

        "mean_abs_shap": mean_abs_shap

    })

    shap_df = (
        shap_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    shap_df.to_csv(

        OUTPUT_DIR
        / "margin_shap_values.csv",

        index=False

    )

    print()
    print("=" * 60)
    print("Top SHAP Features")
    print("=" * 60)
    print()

    print(shap_df.head(20))

    print()
    print("=" * 60)
    print("Files Saved")
    print("=" * 60)
    print()

    print(
        OUTPUT_DIR
        / "margin_feature_importance.csv"
    )

    print(
        OUTPUT_DIR
        / "margin_shap_values.csv"
    )

    print(
        OUTPUT_DIR
        / "margin_shap_summary.png"
    )

    print()


if __name__ == "__main__":

    main()