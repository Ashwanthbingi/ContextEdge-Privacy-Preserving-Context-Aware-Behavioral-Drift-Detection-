from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from contextedge.detection.anomaly_scoring import empirical_pvalue
from contextedge.detection.drift_detection import DriftDetectors
from contextedge.detection.isolation_forest import IsolationDetector
from contextedge.detection.state_machine import DRIFT, NORMAL, TEMPORARY, PersistenceTracker


def test_empirical_rank_conservative_ties():
    assert empirical_pvalue([1, 1, 1], 1) == 1
    assert empirical_pvalue([1, 2, 3], 4) == 0.25
    with pytest.raises(ValueError):
        empirical_pvalue([], 1)


def test_untrained_isolation_rejects_prediction():
    with pytest.raises(RuntimeError):
        IsolationDetector().score(np.zeros((1, 3)))


@pytest.mark.parametrize("mode", ["adwin", "page_hinkley", "either", "both"])
def test_real_drift_algorithms_confirm_sustained_shift(settings, mode):
    settings = replace(settings, drift_confirmation=mode)
    detectors = DriftDetectors(settings, [0.05] * 150)
    signals = [detectors.update(0.99) for _ in range(120)]
    assert any(s["confirmed"] for s in signals)
    assert any(s["adwin"] for s in signals)
    assert any(s["page_hinkley"] for s in signals)


def test_isolated_spike_recovers_without_drift(settings):
    tracker = PersistenceTracker(settings, [0.1] * 150)
    assert tracker.update(0, 0.1, False).classification == NORMAL
    decision = tracker.update(2, 0.99, True)
    assert decision.classification == TEMPORARY
    recovery = tracker.update(4, 0.1, False)
    assert recovery.classification == NORMAL
    assert "temporary" in recovery.explanation


def test_sustained_anomaly_becomes_drift(settings):
    tracker = PersistenceTracker(settings, [0.1] * 150)
    decisions = [tracker.update(i * 2, 0.99, True) for i in range(120)]
    assert all(d.classification != DRIFT for d in decisions[: settings.min_persistence - 1])
    assert any(d.classification == DRIFT for d in decisions)
    assert decisions[-1].persistence_seconds == 238


def test_grace_period_and_monotonic_time(settings):
    tracker = PersistenceTracker(settings, [0.1] * 100)
    for i in range(30):
        assert tracker.update(i, 0.99, True, suppress=True).classification == NORMAL
    assert tracker.count == 0
    with pytest.raises(ValueError):
        tracker.update(20, 0.5, False)
