from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contextedge.collector.telemetry import TelemetrySample
from contextedge.config import Context
from contextedge.context.mappings import CATEGORIES
from contextedge.features.extraction import FeatureExtractor

# These means generate labelled TEST data only; the live engine never imports this module.
SYNTHETIC_MEANS = {
    Context.IDLE: (3, 34, 0.05, 0.01),
    Context.CODING: (27, 48, 1.2, 0.3),
    Context.BROWSING: (16, 52, 0.3, 2.0),
    Context.MEDIA: (19, 43, 0.2, 0.7),
    Context.COMPUTE: (66, 64, 3.0, 0.1),
    Context.UNKNOWN: (20, 45, 0.4, 0.4),
}
CATEGORY = {
    Context.CODING: "coding",
    Context.BROWSING: "browsing",
    Context.MEDIA: "media",
    Context.COMPUTE: "compute",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    context: Context
    change: str | None = None
    duration: int = 0
    before: int = 100
    after: int = 100
    persistent: bool = False
    adapt: bool = False


SCENARIOS = [
    Scenario("normal_coding", Context.CODING),
    Scenario("normal_browsing", Context.BROWSING),
    Scenario("normal_media", Context.MEDIA),
    Scenario("normal_idle", Context.IDLE),
    Scenario("normal_high_compute", Context.COMPUTE),
    Scenario("short_cpu_spike", Context.CODING, "cpu", 3),
    Scenario("short_memory_spike", Context.CODING, "memory", 3),
    Scenario("temporary_disk_network", Context.BROWSING, "io", 3),
    Scenario("sustained_cpu_change", Context.CODING, "cpu", 100, persistent=True),
    Scenario("sustained_memory_change", Context.CODING, "memory", 100, persistent=True),
    Scenario("sudden_context_profile_change", Context.MEDIA, "all", 100, persistent=True),
    Scenario("legitimate_change_adaptation", Context.CODING, "cpu", 480, persistent=True, adapt=True),
]


def sample_stream(context, count, seed, start=0.0, interval=2.0, change=None, change_start=0, change_end=0):
    rng = np.random.default_rng(seed)
    context = Context(context)
    cpu, memory, disk, network = SYNTHETIC_MEANS[context]
    noise = np.zeros(4)
    for index in range(count):
        noise = 0.55 * noise + rng.normal(0, [1.8, 0.35, 0.08, 0.05])
        c, m, d, n = np.array([cpu, memory, disk, network]) + noise
        changed = change is not None and change_start <= index < change_end
        if changed:
            if change in {"cpu", "all"}:
                c += 49
            if change in {"memory", "all"}:
                m += 24
            if change in {"io", "all"}:
                d += 35
                n += 20
        c, m = float(np.clip(c, 0, 100)), float(np.clip(m, 0, 100))
        d, n = float(max(d, 0)), float(max(n, 0))
        counts = dict.fromkeys(CATEGORIES, 0)
        counts["other"] = 100
        counts[CATEGORY.get(context, "other")] += 25
        activity = dict.fromkeys(CATEGORIES, 0.0)
        activity[CATEGORY.get(context, "other")] = c * 0.9
        metrics = {
            "cpu_percent": c,
            "ram_percent": m,
            "available_gb": 16 * (1 - m / 100),
            "core_max": min(100, c + 10),
            "load_per_core": c / 100,
            "disk_read_mb_s": d * 0.6,
            "disk_write_mb_s": d * 0.4,
            "net_recv_mb_s": n * 0.8,
            "net_sent_mb_s": n * 0.2,
            "process_count": 125 + float(rng.integers(-2, 3)),
            "process_cpu": c * 0.9,
            "process_memory": m * 0.85,
            "process_cpu_coverage": 1.0,
        }
        yield TelemetrySample(start + index * interval, metrics, counts, activity, None, interval), changed


def calibration_rows(context, count, settings, seed=42, start=0.0):
    extractor = FeatureExtractor(settings.feature_window)
    rows = []
    for sample, _ in sample_stream(context, count + settings.feature_window - 1, seed, start):
        feature = extractor.extract(sample)
        if feature.valid and feature.window_ready:
            rows.append(
                {
                    "timestamp": sample.timestamp,
                    "features": feature.vector,
                    "segment": f"synthetic-{context}-{seed}",
                    "origin": "synthetic_normal",
                }
            )
    return rows
