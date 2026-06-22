from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_v4.parquet"
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
    / "xgboost_baseline.pkl"
)


def train_model():

    df = pd.read_parquet(INPUT_FILE)

    features = [
    "elo_diff",
    "grass_elo_diff",
    "rank_diff",
    "career_win_pct_diff",
    "grass_win_pct_diff",
    "form_diff_5",
    "form_diff_10",
    "grass_form_diff_10",
    "h2h_win_pct_diff",
    "grass_h2h_win_pct_diff"
    ]

    X = df[features]

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\nModel Results")
    print("-" * 50)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"ROC AUC  : {auc:.4f}"
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print("\nFeature Importance")

    print(importance)

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel Saved: {MODEL_FILE}"
    )


if __name__ == "__main__":
    train_model()