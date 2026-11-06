import json

import pytest

from contextedge.config import Context
from contextedge.evaluation.labelled import evaluate_labelled
from contextedge.evaluation.scenarios import calibration_rows


def test_labelled_evaluation_requires_labels_and_does_not_train(trained_engine, settings, tmp_path):
    path = tmp_path / "labelled.jsonl"
    rows = calibration_rows(Context.CODING, 50, settings, seed=841)
    for row in rows:
        row.update(context="Coding", feature_schema=2, feature_window=4, anomaly=False, drift=False)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    before = len(trained_engine.repo.versions("Coding"))
    results = evaluate_labelled(path, trained_engine.profiles, settings)
    assert results["source"] == "labelled_local"
    assert results["ai"]["observations"] > 0
    assert len(trained_engine.repo.versions("Coding")) == before
    rows[0]["anomaly"] = None
    path.write_text(json.dumps(rows[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit boolean"):
        evaluate_labelled(path, trained_engine.profiles, settings)
