from __future__ import annotations

import pytest

from contextedge.evaluation.metrics import classification_metrics, latency_summary
from contextedge.evaluation.scenarios import SCENARIOS, sample_stream


def test_all_twelve_scenarios_have_reproducible_ground_truth():
    assert len(SCENARIOS) == 12
    for scenario in SCENARIOS:
        args = (scenario.context, scenario.before + scenario.duration + scenario.after, 42)
        kwargs = dict(
            change=scenario.change,
            change_start=scenario.before,
            change_end=scenario.before + scenario.duration,
        )
        first = list(sample_stream(*args, **kwargs))
        second = list(sample_stream(*args, **kwargs))
        assert [s.metrics for s, _ in first] == [s.metrics for s, _ in second]
        assert sum(label for _, label in first) == scenario.duration


def test_confusion_matrix_and_latency_metrics():
    result = classification_metrics([False, False, True, True], [False, True, True, False])
    assert result["precision"] == result["recall"] == result["f1"] == result["false_positive_rate"] == 0.5
    assert latency_summary([])["mean_ms"] is None
    assert latency_summary([1, 2, 3])["p50_ms"] == 2
    with pytest.raises(ValueError):
        classification_metrics([], [])
