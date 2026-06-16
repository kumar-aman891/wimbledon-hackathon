from pathlib import Path
import pandas as pd

from src.elo.elo_utils import (
    expected_score,
    update_rating
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches_for_elo.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_elo.parquet"
)


def build_elo():

    matches = pd.read_parquet(INPUT_FILE)

    overall_ratings = {}

    grass_ratings = {}

    for _, match in matches.iterrows():

        winner = match["winner_id"]
        loser = match["loser_id"]

        surface = str(
            match["surface"]
        ).upper()

        # ------------------
        # Overall Elo
        # ------------------

        winner_rating = (
            overall_ratings
            .get(winner, 1500)
        )

        loser_rating = (
            overall_ratings
            .get(loser, 1500)
        )

        expected_winner = expected_score(
            winner_rating,
            loser_rating
        )

        expected_loser = expected_score(
            loser_rating,
            winner_rating
        )

        overall_ratings[winner] = update_rating(
            winner_rating,
            1,
            expected_winner
        )

        overall_ratings[loser] = update_rating(
            loser_rating,
            0,
            expected_loser
        )

        # ------------------
        # Grass Elo
        # ------------------

        if surface == "GRASS":

            winner_grass = (
                grass_ratings
                .get(winner, 1500)
            )

            loser_grass = (
                grass_ratings
                .get(loser, 1500)
            )

            expected_winner_grass = expected_score(
                winner_grass,
                loser_grass
            )

            expected_loser_grass = expected_score(
                loser_grass,
                winner_grass
            )

            grass_ratings[winner] = update_rating(
                winner_grass,
                1,
                expected_winner_grass
            )

            grass_ratings[loser] = update_rating(
                loser_grass,
                0,
                expected_loser_grass
            )

    players = []

    all_players = set(
        overall_ratings.keys()
    )

    for player_id in all_players:

        
        player_info =    {
                "player_id": player_id,
                "overall_elo":
                    round(
                        overall_ratings[player_id],
                        2
                    ),
                "grass_elo":
                    round(
                        grass_ratings.get(
                            player_id,
                            1500
                        ),
                        2
                    )
            }
        players.append(player_info)
        

    elo_df = pd.DataFrame(players)

    elo_df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(elo_df.shape)

    return elo_df


if __name__ == "__main__":
    build_elo()