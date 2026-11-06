import pytest

from contextedge.config import Context
from contextedge.evaluation.scenarios import sample_stream


@pytest.mark.parametrize("change", ["cpu", "memory", "io"])
def test_short_raw_spikes_do_not_become_drift_through_overlapping_windows(trained_engine, change):
    engine = trained_engine
    engine.set_context("Coding")
    decisions = []
    for sample, _ in sample_stream(Context.CODING, 180, 203, change=change, change_start=70, change_end=73):
        decisions.append(engine.process(sample, origin="synthetic"))
    assert any(r["classification"] == "TEMPORARY ANOMALY" for r in decisions[70:100])
    assert not any(r["classification"] == "PERSISTENT DRIFT" for r in decisions)
    assert decisions[-1]["classification"] == "NORMAL"


def test_missing_sample_requires_fresh_valid_feature_window(trained_engine):
    engine = trained_engine
    engine.set_context("Coding")
    samples = list(sample_stream(Context.CODING, 20, 7))
    for sample, _ in samples[:10]:
        engine.process(sample)
    samples[10][0].metrics["cpu_percent"] = None
    assert not engine.process(samples[10][0])["valid"]
    for sample, _ in samples[11:14]:
        assert engine.process(sample)["anomaly_score"] is None
    assert engine.process(samples[14][0])["anomaly_score"] is not None
