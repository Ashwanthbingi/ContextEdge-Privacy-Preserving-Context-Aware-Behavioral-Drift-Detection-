from __future__ import annotations

import shutil
from dataclasses import replace

import pytest

from contextedge.config import Context, Settings
from contextedge.engine import ContextEdgeEngine
from contextedge.evaluation.scenarios import calibration_rows


@pytest.fixture
def settings():
    return replace(
        Settings(),
        feature_window=4,
        sequence_length=8,
        temporal_epochs=2,
        min_validation_samples=30,
        min_persistence=10,
        page_hinkley_threshold=3.0,
    )


@pytest.fixture(scope="session")
def trained_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("trained")
    cfg = replace(
        Settings(),
        feature_window=4,
        sequence_length=8,
        temporal_epochs=2,
        min_validation_samples=30,
        min_persistence=10,
        page_hinkley_threshold=3.0,
    )
    engine = ContextEdgeEngine(root, cfg)
    for context in (Context.CODING, Context.IDLE):
        engine.profiles.train(context, calibration_rows(context, 300, cfg, seed=42))
    return root


@pytest.fixture
def trained_engine(tmp_path, trained_store, settings):
    destination = tmp_path / "store"
    shutil.copytree(trained_store, destination)
    return ContextEdgeEngine(destination, settings)
