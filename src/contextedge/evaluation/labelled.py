from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

from contextedge.config import Context, Settings
from contextedge.detection.state_machine import DRIFT, PersistenceTracker
from contextedge.evaluation.metrics import classification_metrics, latency_summary
from contextedge.features.extraction import FEATURE_SCHEMA
from contextedge.features.preprocessing import validate_matrix
from contextedge.profiles.manager import ProfileManager


def evaluate_labelled(path: Path, profiles: ProfileManager, settings: Settings) -> dict:
    """Evaluate frozen profiles on explicitly labelled numeric JSONL; never train."""
    decisions = []
    history = None
    sequence = deque()
    tracker = None
    last_timestamp = None
    grace = 0
    provenance = {}
    with Path(path).open(encoding="utf-8") as source:
        for index, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("feature_schema") != FEATURE_SCHEMA:
                raise ValueError(f"Line {index}: incompatible feature schema")
            context = str(Context(row["context"]))
            if type(row.get("anomaly")) is not bool or type(row.get("drift")) is not bool:
                raise ValueError(f"Line {index}: explicit boolean anomaly and drift labels required")
            if row["drift"] and not row["anomaly"]:
                raise ValueError(f"Line {index}: persistent drift must also be labelled anomalous")
            vector = validate_matrix([row["features"]])[0]
            timestamp = float(row["timestamp"])
            if last_timestamp is not None and timestamp <= last_timestamp:
                raise ValueError("Labelled timestamps must strictly increase")
            profile = profiles.active(context)
            if profile is None:
                raise ValueError(f"No ready frozen profile for {context}")
            if row.get("feature_window") != profile.metadata["feature_window"]:
                raise ValueError("Labelled feature-window configuration does not match training")
            provenance[context] = {
                "version": profile.version,
                "training_origins": profile.metadata["origins"],
            }
            key = (context, row["segment"], profile.version)
            gap = last_timestamp is not None and timestamp - last_timestamp > settings.max_context_gap_seconds
            if key != history or gap:
                sequence = deque(maxlen=profile.metadata["sequence_length"])
                footprint = profile.metadata["feature_window"] + profile.metadata["sequence_length"] - 1
                tracker = PersistenceTracker(
                    settings, profile.metadata["reference_scores"], receptive_field=footprint
                )
                history, grace = key, settings.context_grace
            last_timestamp = timestamp
            sequence.append(vector)
            start = time.perf_counter()
            result = profile.score(vector, list(sequence), settings.false_alarm_alpha)
            latency = (time.perf_counter() - start) * 1000
            decision = tracker.update(timestamp, result.score, result.anomalous, suppress=grace > 0)
            grace = max(0, grace - 1)
            if grace or len(sequence) < sequence.maxlen:
                continue
            decisions.append(
                {
                    "timestamp": timestamp,
                    "context": context,
                    "segment": row["segment"],
                    "truth": row["anomaly"],
                    "drift_truth": row["drift"],
                    "prediction": result.anomalous,
                    "drift_prediction": decision.classification == DRIFT,
                    "classification": decision.classification,
                    "score": result.score,
                    "inference_ms": latency,
                }
            )
    if not decisions:
        raise ValueError("No eligible labelled observations after sequence warm-up")
    negatives = [r for r in decisions if not r["drift_truth"]]
    # Delays are per labelled contiguous drift interval. Undetected intervals remain null.
    delays, onset = [], None
    for row in decisions:
        key = (row["context"], row["segment"])
        if row["drift_truth"]:
            if onset is None or onset["key"] != key:
                if onset:
                    delays.append(onset["delay"])
                onset = {"key": key, "time": row["timestamp"], "delay": None}
            if row["drift_prediction"] and onset["delay"] is None:
                onset["delay"] = row["timestamp"] - onset["time"]
        elif onset:
            delays.append(onset["delay"])
            onset = None
    if onset:
        delays.append(onset["delay"])
    return {
        "source": "labelled_local",
        "name": "Frozen-profile labelled local evaluation",
        "created_at": time.time(),
        "training_provenance": provenance,
        "ai": classification_metrics([r["truth"] for r in decisions], [r["prediction"] for r in decisions]),
        "false_drift_observation_rate": sum(r["drift_prediction"] for r in negatives) / len(negatives)
        if negatives
        else None,
        "detection_delays_seconds": delays,
        "inference_latency": latency_summary([r["inference_ms"] for r in decisions]),
        "decisions": decisions,
        "limitations": [
            "Labels are supplied by the experimenter and are not independently verified.",
            "Test source must be independent of calibration; the caller is responsible for avoiding session overlap.",
            "Initial per-segment warm-up is excluded; delays start at the first eligible positive label.",
        ],
    }
