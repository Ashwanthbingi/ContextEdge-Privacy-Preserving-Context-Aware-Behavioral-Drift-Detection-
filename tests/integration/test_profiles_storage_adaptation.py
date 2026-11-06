from __future__ import annotations

import json
import time

import numpy as np
import pytest

from contextedge.config import Context
from contextedge.engine import ContextEdgeEngine
from contextedge.evaluation.scenarios import calibration_rows, sample_stream
from contextedge.features.extraction import FEATURE_NAMES
from contextedge.profiles.calibration import split_calibration, stability_score
from contextedge.storage.database import Database


def test_insufficient_and_discontinuous_calibration(settings):
    with pytest.raises(ValueError, match="Need"):
        split_calibration([], settings)
    rows = calibration_rows(Context.CODING, 260, settings)
    for i, row in enumerate(rows):
        row["segment"] = str(i)
    with pytest.raises(ValueError, match="contiguous"):
        split_calibration(rows, settings)


def test_trained_models_and_onnx_parity(trained_engine):
    profile = trained_engine.profiles.active(Context.CODING)
    assert profile is not None and profile.isolation.ready
    assert profile.metadata["onnx_parity"]["passed"]
    assert profile.metadata["onnx_parity"]["max_absolute_error"] < 1e-5
    assert profile.metadata["validation_sequences"] >= 30
    assert profile.temporal.session.get_providers() == ["CPUExecutionProvider"]


def test_anomaly_scoring_uses_learned_profile(trained_engine, settings):
    profile = trained_engine.profiles.active(Context.CODING)
    normal = calibration_rows(Context.CODING, 60, settings, seed=81)
    seq = [r["features"] for r in normal[-settings.sequence_length :]]
    normal_scores = [
        profile.score(
            normal[i]["features"], [r["features"] for r in normal[i - 7 : i + 1]], settings.false_alarm_alpha
        ).score
        for i in range(7, len(normal))
    ]
    shifted = np.array(seq)
    shifted[:, FEATURE_NAMES.index("cpu_percent")] = 99
    shifted[:, FEATURE_NAMES.index("ram_percent")] = 99
    abnormal = profile.score(shifted[-1], shifted, settings.false_alarm_alpha)
    assert abnormal.anomalous and abnormal.score > np.median(normal_scores)
    assert "GRU/ONNX" in abnormal.method
    warm = profile.score(seq[-1], [], settings.false_alarm_alpha)
    assert "sequence warm-up" in warm.method


def test_profile_persists_across_restart(trained_engine, settings):
    other = ContextEdgeEngine(trained_engine.data_dir, settings)
    profile = other.profiles.active(Context.CODING)
    assert profile.version == 1
    assert profile.metadata["context"] == "Coding"


def test_corrupt_artifact_fails_closed(trained_engine):
    path = trained_engine.profiles._path(Context.CODING, 1) / "model.joblib"
    path.write_bytes(b"corrupted")
    trained_engine.profiles.cache.clear()
    assert trained_engine.profiles.active(Context.CODING) is None
    assert "Coding" in trained_engine.profiles.errors


def test_stability_and_reviewed_adaptation_rollback(trained_engine, settings):
    baseline = trained_engine.profiles.active(Context.CODING)
    manager = trained_engine.adaptation
    start = time.time()
    manager.begin("Coding", start)
    with pytest.raises(ValueError, match="Not enough"):
        manager.stage("Coding")
    rows = calibration_rows(Context.CODING, settings.adaptation_samples, settings, seed=97)
    for i, row in enumerate(rows):
        vector = list(row["features"])
        vector[0] += 40
        manager.observe("Coding", start + i * 2, vector)
    assert stability_score([r["features"] for r in rows]) < settings.adaptation_stability_limit
    candidate = manager.stage("Coding")
    assert trained_engine.profiles.active(Context.CODING).version == baseline.version
    assert candidate.version == 2
    adaptation_id = trained_engine.db.rows("SELECT id FROM adaptations")[0]["id"]
    with pytest.raises(ValueError, match="disabled"):
        manager.accept(adaptation_id, automatic=True)
    manager.accept(adaptation_id, timestamp=start + 500)
    assert trained_engine.profiles.active(Context.CODING).version == 2
    assert (
        trained_engine.repo.calibration_count(Context.CODING) == 0
    )  # Quarantine never pollutes calibration.
    trained_engine.profiles.rollback(Context.CODING, 1)
    assert trained_engine.profiles.active(Context.CODING).version == 1
    assert (
        trained_engine.db.rows("SELECT adaptation_seconds FROM adaptations")[0]["adaptation_seconds"] == 500
    )


def test_database_schema_retention_and_settings(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    db.set_setting("settings", {"retention_days": 3})
    assert Database(db.path).get_setting("settings")["retention_days"] == 3
    db.execute("INSERT INTO system_events(timestamp,kind,message) VALUES (0,'test','old')")
    db.system_event("test", "recent")
    db.retain(1)
    assert len(db.rows("SELECT * FROM system_events")) == 1
    db.execute("UPDATE schema_version SET version=999")
    with pytest.raises(RuntimeError, match="Unsupported"):
        Database(db.path)


def test_database_v1_migration_preserves_old_rows(tmp_path):
    db = Database(tmp_path / "migration.sqlite3")
    with db.connect() as conn:
        conn.execute("ALTER TABLE observations DROP COLUMN feature_schema")
        conn.execute("ALTER TABLE calibration_samples DROP COLUMN feature_schema")
        conn.execute("UPDATE schema_version SET version=1")
        conn.execute(
            "INSERT INTO calibration_samples(timestamp,context,features,segment,origin) VALUES (0,'Idle','[1]','old','user_normal')"
        )
    migrated = Database(db.path)
    assert migrated.rows("SELECT version FROM schema_version")[0]["version"] == 2
    assert migrated.rows("SELECT feature_schema FROM calibration_samples")[0]["feature_schema"] == 1


def test_engine_uncalibrated_restart_and_events(tmp_path, settings):
    engine = ContextEdgeEngine(tmp_path, settings)
    engine.set_context("Browsing")
    engine.set_calibration(True)
    for value, _ in sample_stream(Context.BROWSING, 12, 4, time.time()):
        record = engine.process(value)
    assert record["anomaly_score"] is None
    assert "UNCALIBRATED" in record["classification"]
    other = ContextEdgeEngine(tmp_path, settings)
    assert len(other.repo.recent()) == 12
    assert other.repo.calibration_count("Browsing") == 9
    assert not other.calibrating


def test_engine_context_switch_never_bridges_persistence(trained_engine):
    engine = trained_engine
    engine.set_context("Coding")
    for value, _ in sample_stream(Context.CODING, 120, 66, 0, change="all", change_start=30, change_end=120):
        record = engine.process(value, origin="synthetic")
    assert record["classification"] == "PERSISTENT DRIFT"
    engine.set_context("Idle")
    for value, _ in sample_stream(Context.IDLE, 30, 55, 300):
        record = engine.process(value, origin="synthetic")
    assert record["classification"] != "PERSISTENT DRIFT"
    assert record["persistence_seconds"] == 0
    assert engine._sequence.maxlen == engine.settings.sequence_length
    assert all(event["ended_at"] is not None for event in engine.repo.events())


def test_anomalous_calibration_samples_are_excluded(trained_engine):
    engine = trained_engine
    engine.set_context("Coding")
    engine.set_calibration(True)
    records = []
    for value, _ in sample_stream(Context.CODING, 60, 66, 0, change="all", change_start=0, change_end=60):
        records.append(engine.process(value, origin="synthetic"))
    assert engine.repo.calibration_count("Coding") < 10
    assert any("Excluded" in r["explanation"] for r in records)


def test_delete_private_data_and_profiles(trained_engine):
    engine = trained_engine
    engine.set_context("Coding")
    engine.db.system_event("test", "delete me")
    engine.delete_data(False)
    assert not engine.db.rows("SELECT * FROM system_events")
    assert engine.profiles.active(Context.CODING) is not None
    engine.delete_data(True)
    assert engine.manual_context is None
    assert engine.profiles.active(Context.CODING) is None
    assert not list((engine.data_dir / "profiles").rglob("*.onnx"))


def test_storage_only_numerical_features_and_categories(tmp_path, settings):
    engine = ContextEdgeEngine(tmp_path, settings)
    engine.set_context("Coding")
    value = next(sample_stream(Context.CODING, 1, 4, time.time()))[0]
    engine.process(value)
    record = engine.repo.recent()[0]
    assert all(isinstance(x, (int, float)) for x in record["features"])
    serialized = json.dumps(record)
    for forbidden in ("cmdline", "exe_path", "username", "hostname", "url", "ip_address", "window_title"):
        assert forbidden not in serialized
