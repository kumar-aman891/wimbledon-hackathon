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
    / "xgboost_final.pkl"
)


def train_model():

    print("Loading training dataset...")

    df = pd.read_parquet(
        INPUT_FILE
    )

    # ----------------------------------
    # SAFE FEATURES ONLY
    # ----------------------------------

    features = [

        # Strength
        "elo_diff",
        "grass_elo_diff",
        "rank_diff",

        # Career
        "career_win_pct_diff",
        "grass_win_pct_diff",
        "experience_diff",

        # Form
        "form_diff_5",
        "form_diff_10",
        "grass_form_diff_10",

        # H2H
        "h2h_win_pct_diff",
        "grass_h2h_win_pct_diff",

        # Age
        "age_diff",

        # Surface specialization
        "grass_specialization_diff"
    ]

    # ----------------------------------
    # Tournament Features
    # ----------------------------------

    tournament_features = [
        "is_grand_slam",
        "is_masters",
        "is_atp500",
        "is_atp250"
    ]

    for col in tournament_features:

        if col in df.columns:
            features.append(col)

    # ----------------------------------
    # Round Features
    # ----------------------------------

    round_features = sorted(
        [
            c
            for c in df.columns
            if c.startswith("round_")
        ]
    )

    features.extend(
        round_features
    )

    # ----------------------------------
    # Surface Features
    # ----------------------------------

    surface_features = sorted(
        [
            c
            for c in df.columns
            if c.startswith("surface_")
        ]
    )

    features.extend(
        surface_features
    )

    print(
        f"\nFeatures Used: {len(features)}"
    )

    print("\nFeature List")

    for f in features:
        print(f)

    # ----------------------------------
    # Data
    # ----------------------------------

    X = df[features].copy()

    X = X.fillna(0)

    y = df["target"]

    # ----------------------------------
    # Split
    # ----------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # ----------------------------------
    # Model
    # ----------------------------------

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    # ----------------------------------
    # Predictions
    # ----------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(X_test)
        [:, 1]
    )

    # ----------------------------------
    # Metrics
    # ----------------------------------

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

    # ----------------------------------
    # Feature Importance
    # ----------------------------------

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

    print("\nFeature Importance")

    print(
        importance
    )

    # ----------------------------------
    # Save
    # ----------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel Saved: {MODEL_FILE}"
    )

    return model


if __name__ == "__main__":
    train_model()