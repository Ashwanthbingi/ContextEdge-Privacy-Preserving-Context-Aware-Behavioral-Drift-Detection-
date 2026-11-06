from __future__ import annotations

import json
import time

from contextedge.storage.database import Database, encode
from contextedge.features.extraction import FEATURE_SCHEMA


class Repository:
    def __init__(self, database: Database):
        self.db = database

    def observe(self, record: dict):
        columns = (
            "timestamp",
            "context",
            "confidence",
            "source",
            "features",
            "valid",
            "missing_fraction",
            "classification",
            "anomaly_score",
            "drift_score",
            "persistence_seconds",
            "explanation",
            "method",
            "inference_ms",
            "profile_version",
            "feature_schema",
        )
        values = [
            encode(record[k])
            if k == "features"
            else FEATURE_SCHEMA
            if k == "feature_schema"
            else record.get(k)
            for k in columns
        ]
        self.db.execute(
            f"INSERT INTO observations({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", values
        )

    def recent(self, limit=300):
        rows = self.db.rows(
            "SELECT * FROM observations WHERE feature_schema=? ORDER BY id DESC LIMIT ?",
            (FEATURE_SCHEMA, int(limit)),
        )
        for row in rows:
            row["features"] = json.loads(row["features"])
        return rows[::-1]

    def add_calibration(self, context, timestamp, vector, segment, origin="user_normal"):
        if origin not in {"user_normal", "synthetic_normal", "reviewed_adaptation"}:
            raise ValueError("Training samples require an explicit normal-data origin")
        self.db.execute(
            "INSERT INTO calibration_samples(timestamp,context,features,segment,origin,feature_schema) VALUES (?,?,?,?,?,?)",
            (timestamp, str(context), encode(vector), segment, origin, FEATURE_SCHEMA),
        )

    def calibration(self, context):
        rows = self.db.rows(
            "SELECT * FROM calibration_samples WHERE context=? AND feature_schema=? ORDER BY timestamp,id",
            (str(context), FEATURE_SCHEMA),
        )
        return rows

    def calibration_count(self, context):
        return self.db.rows(
            "SELECT COUNT(*) AS n FROM calibration_samples WHERE context=? AND feature_schema=?",
            (str(context), FEATURE_SCHEMA),
        )[0]["n"]

    def open_event(self, record):
        return self.db.execute(
            "INSERT INTO events(started_at,updated_at,context,kind,persistence_seconds,peak_score,explanation,profile_version) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                record["timestamp"],
                record["timestamp"],
                record["context"],
                record["classification"],
                record["persistence_seconds"],
                record["anomaly_score"],
                record["explanation"],
                record["profile_version"],
            ),
        )

    def update_event(self, event_id, record, close=False):
        self.db.execute(
            "UPDATE events SET updated_at=?,ended_at=?,kind=?,persistence_seconds=?,"
            "peak_score=MAX(peak_score,?),explanation=? WHERE id=?",
            (
                record["timestamp"],
                record["timestamp"] if close else None,
                record["classification"],
                record["persistence_seconds"],
                record["anomaly_score"] or 0,
                record["explanation"],
                event_id,
            ),
        )

    def events(self, limit=200):
        return self.db.rows("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))

    def versions(self, context=None):
        if context:
            return self.db.rows(
                "SELECT * FROM model_versions WHERE context=? ORDER BY version DESC", (str(context),)
            )
        return self.db.rows("SELECT * FROM model_versions ORDER BY created_at DESC")

    def add_evaluation(self, source, name, results):
        if source not in {"synthetic", "real_laptop", "labelled_local"}:
            raise ValueError("Evaluation provenance is required")
        self.db.execute(
            "INSERT INTO evaluations(timestamp,source,name,results) VALUES (?,?,?,?)",
            (time.time(), source, name, encode(results)),
        )
