import os


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Config:
    MODEL_PATH = os.getenv("MILK_MODEL_PATH", "models/milk_rf_analyzer_12.joblib")
    MODEL_NAME = os.getenv("MILK_MODEL_NAME", "analyzer_12 Random Forest")

    # Prediction history is a team decision (see README). Off by default.
    ENABLE_HISTORY = _bool("MILK_ENABLE_HISTORY", False)
    DB_PATH = os.getenv("MILK_DB_PATH", "predictions.db")

    # Optional JSON file {"Fat": [0, 10], ...}; enforced only if set, because valid
    # ranges must be confirmed from the dataset documentation first.
    RANGES_PATH = os.getenv("MILK_RANGES_PATH") or None

    CORS_ORIGINS = os.getenv("MILK_CORS_ORIGINS", "*").split(",")
    MAX_BATCH = int(os.getenv("MILK_MAX_BATCH", "1000"))
