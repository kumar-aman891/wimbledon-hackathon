from src.data.load_matches import load_matches

df = load_matches(
    tour="wta",
    start_year=2015,
    end_year=2025
)

print(df.dtypes)

print("\n\nObject Columns\n")

print(
    df.select_dtypes(
        include="object"
    ).columns.tolist()
)