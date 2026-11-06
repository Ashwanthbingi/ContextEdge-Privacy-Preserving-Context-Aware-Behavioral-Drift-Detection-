from __future__ import annotations

import ast
import sqlite3
import time
from pathlib import Path

import pytest

from contextedge.engine import ContextEdgeEngine


def test_monitor_stops_on_database_error(tmp_path, settings, monkeypatch):
    settings.sampling_interval = 0.25
    engine = ContextEdgeEngine(tmp_path, settings)
    monkeypatch.setattr(
        engine.repo, "observe", lambda _: (_ for _ in ()).throw(sqlite3.OperationalError("locked"))
    )
    engine.start()
    time.sleep(0.7)
    engine.stop()
    assert not engine.running
    assert "OperationalError" in engine.error


def test_collector_never_calls_private_content_apis():
    root = Path(__file__).resolve().parents[2] / "src" / "contextedge" / "collector"
    forbidden = {
        "cmdline",
        "exe",
        "open_files",
        "connections",
        "net_connections",
        "environ",
        "username",
        "GetWindowText",
        "GetWindowTextW",
        "screenshot",
        "GetAsyncKeyState",
    }
    for file in root.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not attributes.intersection(forbidden)


def test_only_one_collector_can_write_each_store(tmp_path, settings):
    first = ContextEdgeEngine(tmp_path, settings)
    second = ContextEdgeEngine(tmp_path, settings)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="Another ContextEdge"):
            second.start()
        with pytest.raises(RuntimeError, match="Another ContextEdge"):
            second.delete_data(True)
    finally:
        first.stop()
    second.start()
    second.stop()


def test_engine_with_outbound_network_disabled(tmp_path, settings, monkeypatch):
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("Outbound networking attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    engine = ContextEdgeEngine(tmp_path, settings)
    engine.collector.start()
    time.sleep(0.05)
    record = engine.process(engine.collector.sample())
    engine.collector.stop()
    assert record["features"]


def test_dashboard_real_start_stop_and_empty_states(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CONTEXTEDGE_DATA_DIR", str(tmp_path / "dashboard"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "ContextEdge"
    app.sidebar.selectbox[0].select("Coding").run()
    app.sidebar.checkbox[0].check().run()
    app.sidebar.button[0].click().run()
    time.sleep(2.4)
    app.run()
    assert not app.exception
    assert any(m.value == "RUNNING" for m in app.metric)
    app.sidebar.button[1].click().run()
    assert not app.exception
    assert any(m.value == "STOPPED" for m in app.metric)
    assert any(m.label == "CPU" for m in app.metric)
