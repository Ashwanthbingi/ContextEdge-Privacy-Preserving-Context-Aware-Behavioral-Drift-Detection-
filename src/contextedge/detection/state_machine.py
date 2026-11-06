from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from contextedge.config import Settings
from contextedge.detection.drift_detection import DriftDetectors

NORMAL = "NORMAL"
TEMPORARY = "TEMPORARY ANOMALY"
DRIFT = "PERSISTENT DRIFT"
WARMUP = "WARM-UP / UNCALIBRATED"


@dataclass
class Decision:
    classification: str
    drift_score: float
    persistence_seconds: float
    explanation: str
    adwin: bool = False
    page_hinkley: bool = False


class PersistenceTracker:
    def __init__(self, settings: Settings, reference_scores=(), receptive_field=1):
        self.settings = settings
        # One raw spike influences multiple feature/sequence windows. Require new
        # sustained evidence beyond that known footprint before declaring drift.
        self.required_count = settings.min_persistence + max(0, receptive_field - 1)
        self.detectors = DriftDetectors(settings, reference_scores)
        self.flags = deque(maxlen=settings.persistence_window)
        self.started = None
        self.last = None
        self.duration = 0.0
        self.count = 0
        self.confirmed = False

    def update(self, timestamp: float, score: float, anomalous: bool, suppress=False) -> Decision:
        if self.last is not None and timestamp <= self.last:
            raise ValueError("Detection timestamps must increase")
        elapsed = timestamp - self.last if self.last is not None else 0
        self.last = timestamp
        if suppress:
            return Decision(NORMAL, 0.0, 0.0, "Context/model transition grace period; alerts suppressed")
        signals = self.detectors.update(score)
        self.flags.append(bool(anomalous))
        if not anomalous:
            recovered = self.started is not None
            previous = "persistent drift" if self.confirmed else "temporary anomaly"
            self.started, self.duration, self.count, self.confirmed = None, 0.0, 0, False
            self.flags.clear()
            self.detectors.clear_confirmation()
            return Decision(
                NORMAL,
                0.0,
                0.0,
                f"Recovered after {previous}" if recovered else "Within calibrated context baseline",
            )
        if self.started is None:
            self.started = timestamp
            self.duration = 0.0
        else:
            self.duration += min(elapsed, self.settings.max_context_gap_seconds)
        self.count += 1
        ratio = sum(self.flags) / len(self.flags)
        persistent = self.count >= self.required_count and ratio >= self.settings.persistence_ratio
        self.confirmed = self.confirmed or (persistent and signals["confirmed"])
        progress = min(self.count / self.required_count, 1.0) * ratio
        drift_score = progress * (1.0 if signals["confirmed"] or self.confirmed else 0.5)
        detectors = ", ".join(k for k in ("adwin", "page_hinkley") if signals[k]) or "none"
        if self.confirmed:
            return Decision(
                DRIFT,
                drift_score,
                self.duration,
                f"Sustained context deviation ({self.count} observations); drift confirmation: {detectors}. "
                "Behavioural change is not evidence of malicious activity.",
                signals["adwin"],
                signals["page_hinkley"],
            )
        return Decision(
            TEMPORARY,
            drift_score,
            self.duration,
            f"Above calibrated threshold; persistence {self.count}/{self.required_count} (includes temporal footprint), "
            f"confirmation: {detectors}. Provisional until recovery or confirmed drift.",
            signals["adwin"],
            signals["page_hinkley"],
        )
