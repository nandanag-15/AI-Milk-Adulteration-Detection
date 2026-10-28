import json
import sqlite3
from datetime import datetime, timezone


class HistoryStore:
    """Minimal SQLite history. Swap for Postgres etc. once the team decides."""

    def __init__(self, path: str):
        self.path = path
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    input TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities TEXT NOT NULL,
                    model TEXT NOT NULL)"""
            )

    def _conn(self):
        return sqlite3.connect(self.path)

    def add(self, sample: dict, pred: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO predictions (created_at,input,label,confidence,probabilities,model)"
                " VALUES (?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(sample),
                    pred["label"],
                    pred["confidence"],
                    json.dumps(pred["probabilities"]),
                    pred["model"],
                ),
            )

    def list(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,created_at,input,label,confidence,probabilities,model"
                " FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            {
                "id": r[0], "created_at": r[1], "input": json.loads(r[2]),
                "label": r[3], "confidence": r[4],
                "probabilities": json.loads(r[5]), "model": r[6],
            }
            for r in rows
        ]
