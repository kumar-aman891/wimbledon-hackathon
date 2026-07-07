from pathlib import Path

import joblib
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split


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


TARGET = "winning_margin_gt1"


def main():

    print("Loading training dataset...")

    df = pd.read_parquet(DATA_FILE)

    print()
    print("Matches:", len(df))

    X = df[FEATURES]

    y = df[TARGET]

    print()
    print("Target Distribution")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    model = CatBoostClassifier(

    iterations=1000,

    depth=7,

    learning_rate=0.03,

    loss_function="Logloss",

    eval_metric="AUC",

    auto_class_weights="Balanced",

    random_seed=42,

    verbose=100

    )

    print()
    print("Training CatBoost Margin Model...")

    model.fit(

    X_train,
    y_train,

    eval_set=(X_test, y_test),

    use_best_model=True,

    early_stopping_rounds=100

    )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    print()
    print("Margin Model Results")
    print("-" * 50)

    print(
        f"Accuracy : {accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"ROC AUC  : {roc_auc_score(y_test, probabilities):.4f}"
    )

    print()

    print(

        classification_report(

            y_test,

            predictions

        )

    )

    importance = pd.DataFrame(

        {

            "feature": FEATURES,

            "importance": model.get_feature_importance()

        }

    )

    importance = (

        importance

        .sort_values(

            "importance",

            ascending=False

        )

    )

    print()

    print("Top Features")

    print(

        importance.head(20)

    )

    MODEL_FILE.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    joblib.dump(

        model,

        MODEL_FILE

    )

    print()

    print("Saved model to")

    print(MODEL_FILE)
    importance.to_csv(

    PROJECT_ROOT
    / "outputs"
    / "margin_feature_importance.csv",

    index=False

    )


if __name__ == "__main__":
    main()