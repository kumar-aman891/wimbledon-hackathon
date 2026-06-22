from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

from lightgbm import LGBMClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_final.parquet"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "lightgbm_final.pkl"
)


def train_model():

    df = pd.read_parquet(INPUT_FILE)

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

    X = df[features].fillna(0)

    y = df["target"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    probabilities = (
        model.predict_proba(X_test)
        [:, 1]
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\nLightGBM Results")
    print("-" * 50)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"ROC AUC  : {auc:.4f}"
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance":
                model.feature_importances_
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print("\nTop Features")

    print(
        importance.head(20)
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel Saved: {MODEL_FILE}"
    )


if __name__ == "__main__":
    train_model()