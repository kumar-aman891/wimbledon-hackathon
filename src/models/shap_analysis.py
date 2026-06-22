from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "catboost_final.pkl"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_final.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "shap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def run_shap():

    print("Loading model...")

    model = joblib.load(
        MODEL_FILE
    )

    print("Loading data...")

    df = pd.read_parquet(
        DATA_FILE
    )

    features = [
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
        "grass_specialization_diff"
    ]

    features += [
        c for c in df.columns
        if c.startswith("round_")
    ]

    features += [
        c for c in df.columns
        if c.startswith("surface_")
    ]

    for col in [
        "is_grand_slam",
        "is_masters",
        "is_atp500",
        "is_atp250"
    ]:
        if col in df.columns:
            features.append(col)

    X = (
        df[features]
        .fillna(0)
        .sample(
            5000,
            random_state=42
        )
    )

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    print("Generating plots...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        bbox_inches="tight"
    )

    plt.close()

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap":
                abs(shap_values).mean(axis=0)
        }
    )

    importance = (
        importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
    )

    importance.to_csv(
        OUTPUT_DIR
        / "shap_importance.csv",
        index=False
    )

    print("\nTop SHAP Features")

    print(
        importance.head(20)
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "shap_summary.png"
    )

    print(
        OUTPUT_DIR
        / "shap_importance.csv"
    )


if __name__ == "__main__":
    run_shap()