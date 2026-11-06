from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import psutil

from contextedge.engine import ContextEdgeEngine
from contextedge.evaluation.metrics import latency_summary


def run_benchmark(engine: ContextEdgeEngine, seconds=30, output: Path | None = None):
    if seconds < 2:
        raise ValueError("Benchmark duration must be at least two seconds")
    if engine.running:
        raise ValueError("Stop monitoring before starting a benchmark")
    process = psutil.Process(os.getpid())
    cpu_before = process.cpu_times()
    rss = []
    system_cpu = []
    first_id = engine.db.rows("SELECT COALESCE(MAX(id),0) n FROM observations")[0]["n"]
    start = time.perf_counter()
    engine.start()
    try:
        while time.perf_counter() - start < seconds:
            time.sleep(min(0.5, max(0, seconds - (time.perf_counter() - start))))
            rss.append(process.memory_info().rss)
            system_cpu.append(psutil.cpu_percent(interval=None))
    finally:
        engine.stop()
    elapsed = time.perf_counter() - start
    cpu_after = process.cpu_times()
    cpu = (cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system) / elapsed * 100
    rows = engine.db.rows("SELECT * FROM observations WHERE id>? ORDER BY id", (first_id,))
    spacings = [b["timestamp"] - a["timestamp"] for a, b in zip(rows, rows[1:])]
    latency = [r["inference_ms"] for r in rows if r["inference_ms"] is not None]
    models = list((engine.data_dir / "profiles").rglob("*.onnx")) + list(
        (engine.data_dir / "profiles").rglob("*.joblib")
    )
    results = {
        "source": "real_laptop",
        "name": "Local monitoring resource benchmark",
        "created_at": time.time(),
        "duration_seconds": elapsed,
        "requested_seconds": seconds,
        "observations": len(rows),
        "sampling_interval_seconds": engine.settings.sampling_interval,
        "observed_mean_interval_seconds": sum(spacings) / len(spacings) if spacings else None,
        "process_cpu_percent_one_core": cpu,
        "process_cpu_percent_machine": cpu / (psutil.cpu_count() or 1),
        "system_cpu_mean_percent": sum(system_cpu) / len(system_cpu),
        "rss_mean_mb": sum(rss) / len(rss) / 1048576,
        "rss_peak_mb": max(rss) / 1048576,
        "inference_latency": latency_summary(latency),
        "deployment_model_bytes": sum(p.stat().st_size for p in models),
        "logical_cores": psutil.cpu_count(),
        "platform": platform.system(),
        "python": platform.python_version(),
        "error": engine.error,
        "limitations": [
            "Short local run; resource usage depends on other laptop workloads.",
            "CPU is process user+system time; machine-normalized CPU divides by logical cores.",
            "RSS includes loaded Python, Streamlit-independent engine and ML libraries.",
            "No labelled real behaviour: precision, recall and drift accuracy are not measured.",
            "No inference latency is reported unless an actually trained context model was used.",
        ],
    }
    if rows:
        latest = rows[-1]
        results["last_observation"] = {
            "context": latest["context"],
            "classification": latest["classification"],
        }
    engine.repo.add_evaluation("real_laptop", results["name"], results)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    return results
