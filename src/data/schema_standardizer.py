def standardize_schema(df):

    string_columns = [
        "winner_seed",
        "loser_seed",
        "winner_entry",
        "loser_entry",
        "winner_hand",
        "loser_hand",
        "score",
        "round"
    ]

    for col in string_columns:

        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
            )

    return df