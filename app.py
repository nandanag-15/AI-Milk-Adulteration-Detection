"""
Milk Adulteration Detection - single-file Flask backend.

Run (dev):        python app.py
Run (production): gunicorn -w 2 -b 0.0.0.0:8000 app:app
Install:          pip install flask flask-cors joblib pandas scikit-learn gunicorn

Output is a preliminary ML classification, not a definitive food-safety determination.
"""
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ----------------------------------------------------------------------------
# Config (environment variables)
# ----------------------------------------------------------------------------
MODEL_PATH = os.getenv("MILK_MODEL_PATH", "models/milk_rf_analyzer_12.joblib")
MODEL_NAME = os.getenv("MILK_MODEL_NAME", "analyzer_12 Random Forest")
ENABLE_HISTORY = os.getenv("MILK_ENABLE_HISTORY", "true").lower() in ("1", "true", "yes", "on")
DB_PATH = os.getenv("MILK_DB_PATH", "milk_sqlite.db")
RANGES_PATH = os.getenv("MILK_RANGES_PATH") or None   # JSON like {"Fat": [0, 10]}
CORS_ORIGINS = os.getenv("MILK_CORS_ORIGINS", "*").split(",")
MAX_BATCH = int(os.getenv("MILK_MAX_BATCH", "1000"))
PORT = int(os.getenv("PORT", "8000"))

FEATURES = [
    "Cells", "QValue", "Fat", "Protein", "Lactose", "Solids",
    "FFA", "Citrate", "FrzPoint", "SNF", "MUN", "Casein",
]
EXPECTED_LABELS = {"raw", "bicarbonate", "formaldehyde", "peroxide", "starch", "sucrose"}
DISCLAIMER = "Preliminary ML classification only. Not a definitive food-safety determination."


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class ModelService:
    def __init__(self, path, name, ranges_path=None):
        obj = joblib.load(path)
        saved_features = None
        if isinstance(obj, dict) and "model" in obj:      # {"model": ..., "features": [...]}
            saved_features = obj.get("features") or obj.get("feature_names")
            obj = obj["model"]
        self.model = obj

        names = getattr(self.model, "feature_names_in_", None)
        self.features = list(names) if names is not None else list(saved_features or FEATURES)
        if set(self.features) != set(FEATURES):
            raise RuntimeError(f"Model features {self.features} != expected {FEATURES}")

        self.labels = [str(c) for c in self.model.classes_]
        unknown = set(self.labels) - EXPECTED_LABELS
        if unknown:
            raise RuntimeError(f"Model returns unexpected labels: {unknown}")

        self.name = name
        self.ranges = json.loads(Path(ranges_path).read_text()) if ranges_path else {}

    def check_ranges(self, row):
        return [
            f"{f}={row[f]} outside valid range [{lo}, {hi}]"
            for f, (lo, hi) in self.ranges.items()
            if f in row and not (lo <= row[f] <= hi)
        ]

    def predict(self, rows):
        X = pd.DataFrame(rows)[self.features]             # training column order
        out = []
        for p in self.model.predict_proba(X):
            probs = {l: round(float(x), 6) for l, x in zip(self.labels, p)}
            label = max(probs, key=probs.get)
            out.append({"label": label, "confidence": probs[label],
                        "probabilities": probs, "model": self.name,
                        "disclaimer": DISCLAIMER})
        return out


# ----------------------------------------------------------------------------
# History (SQLite; swap once the team decides on a database)
# ----------------------------------------------------------------------------

class HistoryStore:
    def __init__(self, path):
        self.path = path

        # Confirm that the existing database has the expected table.
        with sqlite3.connect(self.path) as c:
            columns = {
                row[1] for row in
                c.execute("PRAGMA table_info(predictions)").fetchall()
            }

        required = {
            "id", "created_at", "model_name", "model_version",
            "Cells", "QValue", "Fat", "Protein", "Lactose",
            "Solids", "FFA", "Citrate", "FrzPoint", "SNF",
            "MUN", "Casein", "label", "confidence", "probabilities"
        }

        missing = required - columns
        if missing:
            raise RuntimeError(
                f"Database is missing required columns: {sorted(missing)}"
            )

    def add(self, sample, pred):
        with sqlite3.connect(self.path) as c:
            c.execute(
                """
                INSERT INTO predictions (
                    model_name, model_version,
                    Cells, QValue, Fat, Protein, Lactose, Solids,
                    FFA, Citrate, FrzPoint, SNF, MUN, Casein,
                    label, confidence, probabilities
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pred["model"],
                    pred.get("model_version"),
                    sample["Cells"],
                    sample["QValue"],
                    sample["Fat"],
                    sample["Protein"],
                    sample["Lactose"],
                    sample["Solids"],
                    sample["FFA"],
                    sample["Citrate"],
                    sample["FrzPoint"],
                    sample["SNF"],
                    sample["MUN"],
                    sample["Casein"],
                    pred["label"],
                    pred["confidence"],
                    json.dumps(pred["probabilities"])
                )
            )

    def list(self, limit=50, offset=0):
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                """
                SELECT *
                FROM predictions
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            ).fetchall()

        results = []
        for row in rows:
            item = dict(row)
            item["input"] = {
                feature: item[feature]
                for feature in (
                    "Cells", "QValue", "Fat", "Protein", "Lactose",
                    "Solids", "FFA", "Citrate", "FrzPoint", "SNF",
                    "MUN", "Casein"
                )
            }
            item["model"] = item["model_name"]
            item["probabilities"] = json.loads(
                item["probabilities"] or "{}"
            )
            results.append(item)

        return results

# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate_sample(data):
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
        if f in data:
            v = data[f]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                errors.append(f"{f} must be a finite number.")
            else:
                clean[f] = float(v)
    return (None, errors) if errors else (clean, [])


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)

model = ModelService(MODEL_PATH, MODEL_NAME, RANGES_PATH)
history = HistoryStore(DB_PATH) if ENABLE_HISTORY else None


def error(status, message, details=None):
    # The frontend shows `message || error`, so put the details in `message`.
    body = {"error": message, "message": message}
    if details:
        body["details"] = details
        body["message"] = f"{message}: " + "; ".join(details)
    return jsonify(body), status


def parse(samples):
    rows, errors = [], []
    for i, raw in enumerate(samples):
        clean, errs = validate_sample(raw)
        if errs:
            errors += [f"sample {i}: {e}" for e in errs]
            continue
        errors += [f"sample {i}: {e}" for e in model.check_ranges(clean)]
        rows.append(clean)
    return (None, error(422, "Validation failed", errors)) if errors else (rows, None)


def run_prediction(rows):
    preds = model.predict(rows)
    if history:
        for r, p in zip(rows, preds):
            history.add(r, p)
    return preds


@app.get("/health")
def health():
    return jsonify(status="ok", model=model.name, history_enabled=history is not None)


@app.get("/features")
def features():
    return jsonify(features=FEATURES, labels=model.labels, ranges_enforced=bool(model.ranges),
                   note="Units and valid ranges must be confirmed from dataset documentation.")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return error(400, "Request body must be valid JSON.")
    rows, err = parse([data])
    if err:
        return err
    return jsonify(run_prediction(rows)[0])


@app.post("/predict/batch")
def predict_batch():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        return error(400, 'Body must be JSON like {"samples": [ {...}, ... ]}.')
    if not 1 <= len(data["samples"]) <= MAX_BATCH:
        return error(422, f"samples must contain 1 to {MAX_BATCH} items.")
    rows, err = parse(data["samples"])
    if err:
        return err
    return jsonify(predictions=run_prediction(rows))


@app.get("/history")
def get_history():
    if not history:
        return error(404, "Prediction history is disabled (MILK_ENABLE_HISTORY=false).")
    limit = min(max(request.args.get("limit", 50, type=int), 1), 500)
    offset = max(request.args.get("offset", 0, type=int), 0)
    return jsonify(history.list(limit, offset))


@app.errorhandler(404)
def not_found(_):
    return error(404, "Not found.")


@app.errorhandler(405)
def method_not_allowed(_):
    return error(405, "Method not allowed.")


@app.errorhandler(500)
def server_error(_):
    return error(500, "Internal server error.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
