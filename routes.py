from flask import Blueprint, current_app, jsonify, request

from .validation import DISCLAIMER, FEATURES, validate_sample

bp = Blueprint("api", __name__)


def _svc():
    return current_app.extensions["milk_model"]


def _history():
    return current_app.extensions.get("milk_history")


def _error(status: int, message, details=None):
    # The frontend shows `message || error`, so put the details in `message`.
    body = {"error": message, "message": message}
    if details:
        body["details"] = details
        body["message"] = f"{message}: " + "; ".join(details)
    return jsonify(body), status


def _parse(samples_raw):
    """Validate samples (+ optional ranges). Returns (rows, error_response)."""
    rows, errors = [], []
    for i, raw in enumerate(samples_raw):
        clean, errs = validate_sample(raw)
        if errs:
            errors += [f"sample {i}: {e}" for e in errs]
            continue
        errors += [f"sample {i}: {e}" for e in _svc().check_ranges(clean)]
        rows.append(clean)
    if errors:
        return None, _error(422, "Validation failed", errors)
    return rows, None


def _store(rows, preds):
    h = _history()
    if h:
        for r, p in zip(rows, preds):
            h.add(r, p)


@bp.get("/health")
def health():
    return jsonify(status="ok", model=_svc().name, history_enabled=_history() is not None)


@bp.get("/features")
def features():
    m = _svc()
    return jsonify(
        features=FEATURES,
        labels=m.labels,
        ranges_enforced=bool(m.ranges),
        note="Units and valid ranges must be confirmed from dataset documentation.",
    )


@bp.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return _error(400, "Request body must be valid JSON.")
    rows, err = _parse([data])
    if err:
        return err
    preds = _svc().predict(rows)
    _store(rows, preds)
    return jsonify({**preds[0], "disclaimer": DISCLAIMER})


@bp.post("/predict/batch")
def predict_batch():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        return _error(400, 'Body must be JSON like {"samples": [ {...}, ... ]}.')
    samples = data["samples"]
    if not 1 <= len(samples) <= current_app.config["MAX_BATCH"]:
        return _error(422, f"samples must contain 1 to {current_app.config['MAX_BATCH']} items.")
    rows, err = _parse(samples)
    if err:
        return err
    preds = _svc().predict(rows)
    _store(rows, preds)
    return jsonify(predictions=[{**p, "disclaimer": DISCLAIMER} for p in preds])


@bp.get("/history")
def history():
    h = _history()
    if not h:
        return _error(404, "Prediction history is disabled (MILK_ENABLE_HISTORY=false).")
    limit = min(max(request.args.get("limit", 50, type=int), 1), 500)
    offset = max(request.args.get("offset", 0, type=int), 0)
    return jsonify(h.list(limit, offset))


@bp.app_errorhandler(404)
def not_found(_):
    return _error(404, "Not found.")


@bp.app_errorhandler(405)
def method_not_allowed(_):
    return _error(405, "Method not allowed.")


@bp.app_errorhandler(500)
def server_error(_):
    return _error(500, "Internal server error.")
