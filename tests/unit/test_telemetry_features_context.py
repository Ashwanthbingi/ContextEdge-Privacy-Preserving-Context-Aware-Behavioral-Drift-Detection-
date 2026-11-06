from __future__ import annotations

import time

import numpy as np
import pytest

from contextedge.collector.platform.foreground import foreground_category
from contextedge.collector.telemetry import TelemetryCollector
from contextedge.config import Context, Settings
from contextedge.context.mappings import DEFAULT_MAPPINGS, categorize, validate_mappings
from contextedge.context.recognition import ContextRecognizer
from contextedge.evaluation.scenarios import sample_stream
from contextedge.features.extraction import FEATURE_NAMES, FeatureExtractor
from contextedge.features.preprocessing import sequences, validate_matrix


def sample(context=Context.CODING):
    return next(sample_stream(context, 1, 42))[0]


def test_collector_real_laptop_start_stop():
    collector = TelemetryCollector()
    with pytest.raises(RuntimeError):
        collector.sample()
    collector.start()
    time.sleep(0.1)
    value = collector.sample()
    collector.stop()
    assert 0 <= value.metrics["cpu_percent"] <= 100
    assert 0 <= value.metrics["ram_percent"] <= 100
    assert value.metrics["process_count"] > 0
    assert value.elapsed > 0
    assert set(value.category_counts) == {"coding", "browsing", "media", "compute", "other"}
    assert not collector.running


def test_missing_platform_metrics(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "disk_io_counters", lambda: None)
    monkeypatch.setattr(psutil, "net_io_counters", lambda: None)
    monkeypatch.setattr(psutil, "getloadavg", lambda: (_ for _ in ()).throw(NotImplementedError()))
    collector = TelemetryCollector()
    collector.start()
    value = collector.sample()
    collector.stop()
    assert value.metrics["disk_read_mb_s"] is None
    assert value.metrics["net_recv_mb_s"] is None
    assert value.metrics["load_per_core"] is None


def test_windows_idle_process_cpu_is_not_counted_as_activity(monkeypatch):
    from types import SimpleNamespace
    import psutil

    values = iter([0.0, 1.0, 2.0])
    cpu_times = [0.0]

    def processes(**kwargs):
        cpu_times[0] += 16
        return [
            SimpleNamespace(
                pid=0,
                info={
                    "pid": 0,
                    "name": "System Idle Process",
                    "create_time": 0,
                    "cpu_times": SimpleNamespace(user=0, system=cpu_times[0]),
                    "memory_percent": 0,
                },
            )
        ]

    monkeypatch.setattr(psutil, "process_iter", processes)
    collector = TelemetryCollector(clock=lambda: next(values))
    collector.start()
    collector.sample()
    assert collector.sample().metrics["process_cpu"] == 0


def test_extraction_rolling_windows_and_trends():
    extractor = FeatureExtractor(4)
    for i in range(4):
        value = sample()
        value.timestamp = i * 2
        value.metrics["cpu_percent"] = 10 + i * 4
        result = extractor.extract(value)
    assert result.window_ready and result.valid
    assert len(result.vector) == len(FEATURE_NAMES)
    assert result.as_dict()["cpu_slope"] == pytest.approx(2.0)
    extractor.reset()
    assert not extractor.extract(value).window_ready


def test_missing_and_nonfinite_features():
    value = sample()
    value.metrics["cpu_percent"] = float("nan")
    result = FeatureExtractor(2).extract(value)
    assert not result.valid
    assert np.isfinite(result.vector).all()
    value = sample()
    value.metrics["disk_read_mb_s"] = None
    result = FeatureExtractor(2).extract(value)
    assert result.as_dict()["disk_available"] == 0
    assert result.missing_fraction > 0


@pytest.mark.parametrize("context", list(Context))
def test_manual_context(context):
    decision = ContextRecognizer().recognize(sample(), str(context))
    assert decision.context == context and decision.confidence == 1 and decision.source == "manual"


@pytest.mark.parametrize(
    "category,expected",
    [
        ("coding", Context.CODING),
        ("browsing", Context.BROWSING),
        ("media", Context.MEDIA),
        ("compute", Context.COMPUTE),
    ],
)
def test_minimal_foreground_context(category, expected):
    value = sample()
    value.foreground = category
    assert ContextRecognizer().recognize(value).context == expected


def test_inferred_context_stability_and_uncertainty():
    recognizer = ContextRecognizer()
    value = sample(Context.CODING)
    assert recognizer.recognize(value).context == Context.UNKNOWN
    recognizer.recognize(value)
    assert recognizer.recognize(value).context == Context.CODING
    value.category_cpu = {"other": 50, "coding": 1, "browsing": 1}
    assert recognizer.recognize(value).context == Context.UNKNOWN


def test_idle_context():
    recognizer = ContextRecognizer()
    value = sample(Context.IDLE)
    for _ in range(3):
        decision = recognizer.recognize(value)
    assert decision.context == Context.IDLE


def test_mapping_paths_and_categories():
    assert categorize("Code.EXE", DEFAULT_MAPPINGS) == "coding"
    assert categorize("private-unknown.exe", DEFAULT_MAPPINGS) == "other"
    with pytest.raises(ValueError):
        validate_mappings({"C:\\private\\name.exe": "coding"})
    with pytest.raises(ValueError):
        validate_mappings({"code": "invalid"})


def test_unsupported_foreground(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    assert foreground_category(DEFAULT_MAPPINGS) is None


def test_sequences_never_bridge_sessions():
    data = np.zeros((8, len(FEATURE_NAMES)), np.float32)
    seq = sequences(data, 4, ["a"] * 4 + ["b"] * 4)
    assert seq.shape == (2, 4, len(FEATURE_NAMES))
    with pytest.raises(ValueError):
        validate_matrix([[1, 2]])


def test_invalid_settings():
    with pytest.raises(ValueError):
        Settings(sampling_interval=0).validate()
    with pytest.raises(ValueError):
        Settings(false_alarm_alpha=0.001).validate()
