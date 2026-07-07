from pathlib import Path

import pandas as pd

from src.prediction.predict_match import predict_match


PROJECT_ROOT = Path(__file__).resolve().parents[2]


FIXTURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "fixtures"
    / "fixtures.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "predicted_fixtures.csv"
)


def main():

    print()

    print("=" * 70)

    print("Loading Fixtures")

    print("=" * 70)

    fixtures = pd.read_csv(FIXTURE_FILE)

    print()

    print(f"Fixtures : {len(fixtures)}")

    predictions = []

    for idx, row in fixtures.iterrows():

        print()

        print("-" * 70)

        print(

            f"Predicting "

            f"{idx+1}/{len(fixtures)}"

        )

        result = predict_match(

            player1_name=row["player1"],

            player2_name=row["player2"],

            surface=row["surface"],

            round_name=row["round"],

            tournament=row["tournament"]

        )

        predictions.append({

            "player1": result["player1"],

            "player2": result["player2"],

            "winner": result["winner"],

            "winner_probability":
                round(
                    result["winner_probability"],
                    4
                ),

            "winning_margin":
                result["margin_prediction"],

            "margin_confidence":
                round(
                    result["margin_confidence"],
                    4
                )

        })


    ########################################################

    prediction_df = pd.DataFrame(

        predictions

    )

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    prediction_df.to_csv(

        OUTPUT_FILE,

        index=False

    )

    print()

    print("=" * 70)

    print("Predictions")

    print("=" * 70)

    print()

    print(

        prediction_df

        .to_string(index=False)

    )

    print()

    print("=" * 70)

    print("Saved")

    print("=" * 70)

    print()

    print(OUTPUT_FILE)

    print()


if __name__ == "__main__":

    main()