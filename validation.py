import math

FEATURES = [
    "Cells", "QValue", "Fat", "Protein", "Lactose", "Solids",
    "FFA", "Citrate", "FrzPoint", "SNF", "MUN", "Casein",
]

DISCLAIMER = "Preliminary ML classification only. Not a definitive food-safety determination."


def validate_sample(data) -> tuple[dict | None, list[str]]:
    """Return (clean_sample, errors). Requires exactly the 12 numeric features."""
    if not isinstance(data, dict):
        return None, ["Sample must be a JSON object."]
    errors = []
    missing = [f for f in FEATURES if f not in data]
    extra = [k for k in data if k not in FEATURES]
    if missing:
        errors.append(f"Missing features: {missing}")
    if extra:
        errors.append(f"Unexpected features: {extra}")

    clean = {}
    for f in FEATURES:
        if f not in data:
            continue
        v = data[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            errors.append(f"{f} must be a finite number.")
        else:
            clean[f] = float(v)
    return (None, errors) if errors else (clean, [])
