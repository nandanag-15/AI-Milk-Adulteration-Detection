import json
from pathlib import Path

import joblib
import pandas as pd

from .validation import FEATURES

EXPECTED_LABELS = {"raw", "bicarbonate", "formaldehyde", "peroxide", "starch", "sucrose"}


class ModelService:
    def __init__(self, model_path: str, model_name: str, ranges_path: str | None = None):
        obj = joblib.load(model_path)
        # Support a saved dict like {"model": ..., "features": [...]}
        if isinstance(obj, dict) and "model" in obj:
            self.model = obj["model"]
            saved_features = obj.get("features") or obj.get("feature_names")
        else:
            self.model = obj
            saved_features = None

        names = getattr(self.model, "feature_names_in_", None)
        self.features = list(names) if names is not None else list(saved_features or FEATURES)
        if set(self.features) != set(FEATURES):
            raise RuntimeError(
                f"Model features {self.features} do not match the expected 12: {FEATURES}"
            )

        self.labels = [str(c) for c in self.model.classes_]
        unknown = set(self.labels) - EXPECTED_LABELS
        if unknown:
            raise RuntimeError(f"Model returns unexpected labels: {unknown}")

        self.name = model_name
        self.ranges: dict[str, list[float]] = {}
        if ranges_path:
            self.ranges = json.loads(Path(ranges_path).read_text())

    def check_ranges(self, row: dict) -> list[str]:
        errors = []
        for feat, (lo, hi) in self.ranges.items():
            v = row.get(feat)
            if v is not None and not (lo <= v <= hi):
                errors.append(f"{feat}={v} outside valid range [{lo}, {hi}]")
        return errors

    def predict(self, rows: list[dict]) -> list[dict]:
        X = pd.DataFrame(rows)[self.features]  # enforce column order the model was trained on
        proba = self.model.predict_proba(X)
        out = []
        for p in proba:
            probs = {lbl: round(float(x), 6) for lbl, x in zip(self.labels, p)}
            label = max(probs, key=probs.get)
            out.append(
                {
                    "label": label,
                    "confidence": probs[label],
                    "probabilities": probs,
                    "model": self.name,
                }
            )
        return out
