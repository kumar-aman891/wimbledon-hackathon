from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tournament_level_master.csv"
)

mapping = pd.DataFrame(
    {
        "tourney_level": [
            "G",
            "M",
            "A",
            "I",
            "P",
            "PM",
            "D",
            "F",
            "O",
            "50+H",
            "35+H"
        ],
        "importance_weight": [
            1.50,
            1.25,
            1.00,
            1.15,
            1.15,
            1.20,
            0.75,
            0.50,
            0.50,
            0.25,
            0.25
        ]
    }
)

mapping.to_csv(
    OUTPUT_FILE,
    index=False
)

print(mapping)