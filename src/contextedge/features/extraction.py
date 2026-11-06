from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from contextedge.collector.telemetry import TelemetrySample
from contextedge.context.mappings import CATEGORIES

BASE_FEATURES = (
    "cpu_percent",
    "ram_percent",
    "available_gb",
    "core_max",
    "load_per_core",
    "disk_read_mb_s",
    "disk_write_mb_s",
    "net_recv_mb_s",
    "net_sent_mb_s",
    "process_count",
    "process_cpu",
    "process_memory",
    "process_cpu_coverage",
)
FEATURE_NAMES = (
    BASE_FEATURES
    + tuple("category_" + c for c in CATEGORIES)
    + (
        "cpu_mean",
        "cpu_std",
        "cpu_slope",
        "ram_slope",
        "disk_mean",
        "network_mean",
        "disk_available",
        "network_available",
        "load_available",
    )
)
FEATURE_SCHEMA = 2


@dataclass
class FeatureObservation:
    timestamp: float
    vector: list[float]
    valid: bool
    missing_fraction: float
    window_ready: bool

    def as_dict(self):
        return dict(zip(FEATURE_NAMES, self.vector, strict=True))


class FeatureExtractor:
    def __init__(self, window: int = 8):
        self.window = window
        self.history = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def extract(self, sample: TelemetrySample) -> FeatureObservation:
        values = np.array(
            [sample.metrics.get(k) if sample.metrics.get(k) is not None else np.nan for k in BASE_FEATURES],
            dtype=np.float64,
        )
        available = np.isfinite(values)
        valid = bool(available[0] and available[1] and available.mean() >= 0.65 and sample.elapsed > 0)
        missing = 1.0 - float(available.mean())
        # Availability flags distinguish unsupported I/O/load from true zero activity.
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        self.history.append((sample.timestamp, values))
        data = np.array([v for _, v in self.history])
        times = np.array([t for t, _ in self.history], dtype=np.float64)
        times -= times.mean()
        denom = float(times @ times)

        def slope(column):
            return float(times @ data[:, column] / denom) if denom > 1e-9 else 0.0

        counts = max(sum(sample.category_counts.values()), 1)
        vector = values.tolist() + [sample.category_counts.get(c, 0) / counts for c in CATEGORIES]
        vector += [
            float(data[:, 0].mean()),
            float(data[:, 0].std()),
            slope(0),
            slope(1),
            float(data[:, 5:7].sum(axis=1).mean()),
            float(data[:, 7:9].sum(axis=1).mean()),
            float(available[5] and available[6]),
            float(available[7] and available[8]),
            float(available[4]),
        ]
        ready = len(self.history) == self.window and valid
        if not valid:
            self.history.clear()
        return FeatureObservation(sample.timestamp, vector, valid, missing, ready)
