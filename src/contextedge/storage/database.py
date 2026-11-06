from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 2
SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS observations (
 id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, context TEXT NOT NULL, feature_schema INTEGER NOT NULL DEFAULT 2,
 confidence REAL NOT NULL, source TEXT NOT NULL, features TEXT NOT NULL,
 valid INTEGER NOT NULL, missing_fraction REAL NOT NULL, classification TEXT NOT NULL,
 anomaly_score REAL, drift_score REAL, persistence_seconds REAL DEFAULT 0,
 explanation TEXT NOT NULL, method TEXT NOT NULL, inference_ms REAL, profile_version INTEGER
);
CREATE INDEX IF NOT EXISTS observation_time ON observations(timestamp);
CREATE INDEX IF NOT EXISTS observation_context ON observations(context,timestamp);
CREATE TABLE IF NOT EXISTS profiles (
 context TEXT PRIMARY KEY, active_version INTEGER, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_versions (
 context TEXT NOT NULL REFERENCES profiles(context), version INTEGER NOT NULL,
 status TEXT NOT NULL, created_at REAL NOT NULL, artifact_dir TEXT NOT NULL,
 metadata TEXT NOT NULL, PRIMARY KEY(context,version)
);
CREATE TABLE IF NOT EXISTS calibration_samples (
 id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, context TEXT NOT NULL,
 features TEXT NOT NULL, segment TEXT NOT NULL, origin TEXT NOT NULL, feature_schema INTEGER NOT NULL DEFAULT 2
);
CREATE INDEX IF NOT EXISTS calibration_context ON calibration_samples(context,timestamp);
CREATE TABLE IF NOT EXISTS events (
 id INTEGER PRIMARY KEY, started_at REAL NOT NULL, updated_at REAL NOT NULL,
 ended_at REAL, context TEXT NOT NULL, kind TEXT NOT NULL,
 persistence_seconds REAL NOT NULL, peak_score REAL NOT NULL,
 explanation TEXT NOT NULL, profile_version INTEGER
);
CREATE INDEX IF NOT EXISTS events_time ON events(started_at);
CREATE TABLE IF NOT EXISTS context_history (
 id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, context TEXT NOT NULL,
 confidence REAL NOT NULL, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS adaptations (
 id INTEGER PRIMARY KEY, context TEXT NOT NULL, started_at REAL NOT NULL,
 updated_at REAL NOT NULL, status TEXT NOT NULL, candidate_version INTEGER,
 previous_version INTEGER, sample_count INTEGER DEFAULT 0, stability REAL,
 adaptation_seconds REAL, reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evaluations (
 id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, source TEXT NOT NULL,
 name TEXT NOT NULL, results TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS system_events (
 id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, kind TEXT NOT NULL, message TEXT NOT NULL
);
"""


def encode(value) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            existing = conn.execute("SELECT name FROM sqlite_master WHERE name='schema_version'").fetchone()
            if existing:
                version = conn.execute("SELECT version FROM schema_version").fetchone()
                if version and version[0] == 1:
                    conn.execute("BEGIN IMMEDIATE")
                    # Existing numerical rows predate coverage feature schema 2.
                    # Preserve them for retention/audit but never mix feature contracts.
                    conn.execute(
                        "ALTER TABLE observations ADD COLUMN feature_schema INTEGER NOT NULL DEFAULT 1"
                    )
                    conn.execute(
                        "ALTER TABLE calibration_samples ADD COLUMN feature_schema INTEGER NOT NULL DEFAULT 1"
                    )
                    conn.execute("UPDATE schema_version SET version=2")
                    conn.commit()
                elif version and version[0] != SCHEMA_VERSION:
                    raise RuntimeError("Unsupported database schema; back up and migrate before opening")
            conn.executescript("BEGIN IMMEDIATE;\n" + SCHEMA + "\nCOMMIT;")
            if not conn.execute("SELECT 1 FROM schema_version").fetchone():
                conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
            conn.execute("PRAGMA user_version=2")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA secure_delete=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def execute(self, sql: str, args=()) -> int:
        with self.connect() as conn:
            cur = conn.execute(sql, args)
            return cur.lastrowid

    def rows(self, sql: str, args=()) -> list[dict]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, args).fetchall()]

    def set_setting(self, key, value):
        self.execute(
            "INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, encode(value)),
        )

    def get_setting(self, key, default=None):
        rows = self.rows("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def system_event(self, kind: str, message: str):
        self.execute(
            "INSERT INTO system_events(timestamp,kind,message) VALUES (?,?,?)", (time.time(), kind, message)
        )

    def retain(self, days: int, now=None):
        cutoff = (now or time.time()) - days * 86400
        with self.connect() as conn:
            for table, column in (
                ("observations", "timestamp"),
                ("calibration_samples", "timestamp"),
                ("events", "started_at"),
                ("context_history", "timestamp"),
                ("system_events", "timestamp"),
                ("evaluations", "timestamp"),
            ):
                conn.execute(f"DELETE FROM {table} WHERE {column} < ?", (cutoff,))

    def clear(self, include_profiles=False):
        tables = [
            "observations",
            "calibration_samples",
            "events",
            "context_history",
            "evaluations",
            "system_events",
        ]
        if include_profiles:
            tables += ["adaptations", "model_versions", "profiles", "settings"]
        with self.connect() as conn:
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
        with self.connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
