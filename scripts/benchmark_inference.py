"""Paced CPU inference benchmark using an explicitly synthetic-trained profile."""

import argparse
import json
import os
import time
from pathlib import Path

import psutil

from contextedge.config import Context
from contextedge.engine import ContextEdgeEngine
from contextedge.evaluation.metrics import latency_summary
from contextedge.evaluation.scenarios import calibration_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path("reports/synthetic/results.json"))
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("reports/inference_benchmark.json"))
    args = parser.parse_args()
    if args.seconds < 2 or args.interval < 0.05:
        parser.error("Use at least 2 seconds and a 0.05-second interval")
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if evaluation["source"] != "synthetic":
        parser.error("This benchmark specifically requires a synthetic evaluation artifact")
    engine = ContextEdgeEngine(Path(evaluation["artifact_directory"]))
    profile = engine.profiles.load("Coding", 1)
    rows = calibration_rows(Context.CODING, 200, engine.settings, seed=9102)
    sequence = [r["features"] for r in rows[: profile.metadata["sequence_length"]]]
    for _ in range(10):
        profile.score(sequence[-1], sequence, engine.settings.false_alarm_alpha)
    process = psutil.Process(os.getpid())
    initial_cpu = process.cpu_times()
    initial_process_time = time.process_time()
    rss, latencies = [], []
    start = time.perf_counter()
    deadline = start
    count = 0
    while time.perf_counter() - start < args.seconds:
        sequence = sequence[1:] + [rows[(count + len(sequence)) % len(rows)]["features"]]
        t = time.perf_counter()
        profile.score(sequence[-1], sequence, engine.settings.false_alarm_alpha)
        latencies.append((time.perf_counter() - t) * 1000)
        rss.append(process.memory_info().rss / 1048576)
        count += 1
        deadline += args.interval
        time.sleep(max(0, min(deadline, start + args.seconds) - time.perf_counter()))
    elapsed = time.perf_counter() - start
    final_cpu = process.cpu_times()
    cpu_delta = final_cpu.user + final_cpu.system - initial_cpu.user - initial_cpu.system
    process_time_delta = time.process_time() - initial_process_time
    # Short regularly spaced bursts can fall below Windows CPU accounting ticks.
    # A zero counter is not evidence of free inference; report it as unresolved.
    cpu = cpu_delta / elapsed * 100 if cpu_delta > 0 else None
    results = {
        "source": "synthetic",
        "name": "Paced local CPU inference resource benchmark",
        "created_at": time.time(),
        "duration_seconds": elapsed,
        "interval_seconds": args.interval,
        "inference_latency": latency_summary(latencies),
        "process_cpu_percent_one_core": cpu,
        "process_cpu_percent_machine": cpu / (psutil.cpu_count() or 1) if cpu is not None else None,
        "cpu_counter_delta_seconds": cpu_delta,
        "stdlib_process_time_delta_seconds": process_time_delta,
        "rss_mean_mb": sum(rss) / len(rss),
        "rss_peak_mb": max(rss),
        "model_bytes": profile.metadata["model_bytes"],
        "onnx_bytes": profile.metadata["onnx_bytes"],
        "profile_context": "Coding",
        "profile_version": profile.version,
        "limitations": [
            "Actual hardware measurements on synthetic-trained models and synthetic inputs.",
            "Four inferences/second by default; excludes collector, SQLite writes, dashboard and training.",
            "Includes Python/ML process memory; not a real-world accuracy experiment.",
            "Zero CPU counter deltas are reported as unavailable, not zero-cost inference; Windows accounting may miss short paced bursts.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
