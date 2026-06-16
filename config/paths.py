from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw"

ATP_DATA = RAW_DATA / "atp"
WTA_DATA = RAW_DATA / "wta"

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

OUTPUTS = PROJECT_ROOT / "outputs"