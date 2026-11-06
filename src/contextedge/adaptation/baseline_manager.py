from __future__ import annotations

import time
from collections import deque

from contextedge.config import Settings
from contextedge.profiles.calibration import stability_score
from contextedge.profiles.manager import ProfileManager


class BaselineManager:
    """Quarantine is separate from approved calibration data and active models."""

    def __init__(self, profiles: ProfileManager, settings: Settings):
        self.profiles, self.settings = profiles, settings
        self.pending: dict[str, dict] = {}

    def begin(self, context: str, timestamp: float):
        if context in self.pending:
            return
        active = self.profiles.active(context)
        if not active:
            return
        existing = self.profiles.repo.db.rows(
            "SELECT id,status FROM adaptations WHERE context=? AND status IN ('collecting','review') ORDER BY id DESC",
            (context,),
        )
        if existing and existing[0]["status"] == "review":
            return
        if existing:
            self.profiles.repo.db.execute(
                "UPDATE adaptations SET status='interrupted',updated_at=? WHERE id=?",
                (timestamp, existing[0]["id"]),
            )
        ident = self.profiles.repo.db.execute(
            "INSERT INTO adaptations(context,started_at,updated_at,status,previous_version,reason) VALUES (?,?,?,?,?,?)",
            (
                context,
                timestamp,
                timestamp,
                "collecting",
                active.version,
                "Confirmed drift; observations quarantined pending stability and review",
            ),
        )
        self.pending[context] = {
            "id": ident,
            "start": timestamp,
            "rows": deque(maxlen=self.settings.adaptation_samples),
        }

    def observe(self, context, timestamp, vector):
        if context not in self.pending:
            return
        item = self.pending[context]
        item["rows"].append(
            {
                "timestamp": timestamp,
                "features": list(vector),
                "segment": f"adapt-{item['id']}",
                "origin": "quarantined_candidate",
            }
        )
        self.profiles.repo.db.execute(
            "UPDATE adaptations SET updated_at=?,sample_count=? WHERE id=?",
            (timestamp, len(item["rows"]), item["id"]),
        )

    def reset_collection(self, context, reason="Context continuity ended"):
        item = self.pending.pop(context, None)
        if item:
            self.profiles.repo.db.execute(
                "UPDATE adaptations SET status='interrupted',updated_at=?,reason=? WHERE id=?",
                (time.time(), reason, item["id"]),
            )

    def stage(self, context):
        if context not in self.pending:
            raise ValueError("No confirmed drift collection for this context")
        item = self.pending[context]
        rows = list(item["rows"])
        if len(rows) < self.settings.adaptation_samples:
            raise ValueError("Not enough quarantined observations")
        stability = stability_score([r["features"] for r in rows])
        if stability > self.settings.adaptation_stability_limit:
            raise ValueError(f"Candidate is not stable ({stability:.3f}); continue collecting")
        candidate = self.profiles.train(context, rows=rows, candidate=True)
        timestamp = rows[-1]["timestamp"]
        self.profiles.repo.db.execute(
            "UPDATE adaptations SET status='review',updated_at=?,candidate_version=?,stability=?,reason=? WHERE id=?",
            (
                timestamp,
                candidate.version,
                stability,
                "Stable candidate trained and ONNX parity verified; awaiting acceptance",
                item["id"],
            ),
        )
        self.pending.pop(context)
        return candidate

    def accept(self, adaptation_id: int, automatic=False, timestamp=None):
        rows = self.profiles.repo.db.rows("SELECT * FROM adaptations WHERE id=?", (adaptation_id,))
        if not rows or rows[0]["status"] != "review":
            raise ValueError("No reviewed candidate available")
        row = rows[0]
        if automatic and not self.settings.auto_adapt:
            raise ValueError("Automatic adaptation is disabled")
        self.profiles.activate(row["context"], row["candidate_version"])
        # Duration includes collection and review, using wall time in live operation.
        now = time.time() if timestamp is None else timestamp
        self.profiles.repo.db.execute(
            "UPDATE adaptations SET status='accepted',updated_at=?,adaptation_seconds=?,reason=? WHERE id=?",
            (
                now,
                max(0, now - row["started_at"]),
                "Accepted after stability check "
                + ("under opt-in automatic policy" if automatic else "and user review"),
                adaptation_id,
            ),
        )

    def reject(self, adaptation_id):
        self.profiles.repo.db.execute(
            "UPDATE adaptations SET status='rejected',updated_at=?,reason=? WHERE id=? AND status='review'",
            (time.time(), "Candidate rejected by user; previous baseline retained", adaptation_id),
        )
