from __future__ import annotations

import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from contextedge.config import Context, Settings
from contextedge.detection.state_machine import DRIFT
from contextedge.engine import ContextEdgeEngine
from contextedge.evaluation.metrics import classification_metrics, latency_summary
from contextedge.evaluation.scenarios import SCENARIOS, calibration_rows, sample_stream


def run_evaluation(output: Path, seed=42, epochs=18) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Isolated database/models; never opens the actual monitoring store.
    run_dir = output / f"run-{time.time_ns()}"
    settings = replace(Settings(), temporal_epochs=epochs, random_seed=seed)
    engine = ContextEdgeEngine(run_dir, settings)
    metadata = {}
    for index, context in enumerate(Context):
        if context == Context.UNKNOWN:
            continue
        profile = engine.profiles.train(context, calibration_rows(context, 400, settings, seed + index * 13))
        metadata[str(context)] = profile.metadata
    baseline_versions = {str(c): engine.profiles.active(c).version for c in Context if c != Context.UNKNOWN}
    decisions, summaries, adaptation_times = [], [], []
    stream_time = time.time() - 200000
    for index, scenario in enumerate(SCENARIOS):
        # Fresh temporal state per experiment; re-use only the same frozen baseline models.
        engine = ContextEdgeEngine(run_dir, settings)
        engine.set_context(str(scenario.context))
        engine.profiles.activate(scenario.context, baseline_versions[str(scenario.context)])
        count = scenario.before + scenario.duration + scenario.after
        detected_at = None
        accepted_at = None
        scenario_rows = []
        start = stream_time
        for offset, (sample, changed) in enumerate(
            sample_stream(
                scenario.context,
                count,
                seed + 1000 + index,
                start,
                settings.sampling_interval,
                scenario.change,
                scenario.before,
                scenario.before + scenario.duration,
            )
        ):
            record = engine.process(sample, origin="synthetic")
            if (
                record["anomaly_score"] is None
                or offset < settings.feature_window + settings.sequence_length + settings.context_grace
            ):
                continue
            if record["classification"] == DRIFT and changed and detected_at is None:
                detected_at = sample.timestamp
            # After explicit simulated review, the legitimate shifted regime becomes normal ground truth.
            target = bool(changed if accepted_at is None else not changed)
            row = {
                "scenario": scenario.name,
                "offset": offset,
                "timestamp": sample.timestamp,
                "context": str(scenario.context),
                "ground_truth_anomaly": target,
                "ground_truth_drift": bool(target and scenario.persistent),
                "prediction_anomaly": record["anomaly_score"] >= 1 - settings.false_alarm_alpha,
                "prediction_drift": record["classification"] == DRIFT,
                "classification": record["classification"],
                "anomaly_score": record["anomaly_score"],
                "drift_score": record["drift_score"],
                "inference_ms": record["inference_ms"],
                "method": record["method"],
                "adapted": accepted_at is not None,
            }
            scenario_rows.append(row)
            if scenario.adapt and accepted_at is None:
                pending = engine.adaptation.pending.get(str(scenario.context))
                if pending and len(pending["rows"]) >= settings.adaptation_samples:
                    try:
                        engine.adaptation.stage(str(scenario.context))
                        engine.adaptation.accept(pending["id"], timestamp=sample.timestamp)
                        accepted_at = sample.timestamp
                        adaptation_times.append(sample.timestamp - pending["start"])
                        engine._trackers.clear()
                        engine._sequence.clear()
                    except ValueError:
                        pass
        decisions.extend(scenario_rows)
        metrics = classification_metrics(
            [r["ground_truth_anomaly"] for r in scenario_rows],
            [r["prediction_anomaly"] for r in scenario_rows],
        )
        onset = start + scenario.before * settings.sampling_interval
        false_drift = sum(r["prediction_drift"] and not r["ground_truth_drift"] for r in scenario_rows)
        eligible = sum(not r["ground_truth_drift"] for r in scenario_rows)
        summaries.append(
            {
                "scenario": scenario.name,
                "context": str(scenario.context),
                "ai": metrics,
                "persistent_change_expected": scenario.persistent,
                "detection_delay_seconds": detected_at - onset if detected_at is not None else None,
                "false_drift_observation_rate": false_drift / eligible if eligible else None,
                "false_drift_observations": false_drift,
                "has_false_drift": false_drift > 0,
                "adaptation_accepted": accepted_at is not None,
                "adaptation_time_simulated_seconds": accepted_at - detected_at
                if accepted_at is not None and detected_at is not None
                else None,
            }
        )
        stream_time += (count + 100) * settings.sampling_interval
    frame = pd.DataFrame(decisions)
    frame.to_csv(output / "decisions.csv", index=False)
    # Report adaptation separately: return to the original distribution after adaptation is another change.
    stationary = frame[frame["scenario"] != "legitimate_change_adaptation"]
    no_drift = stationary[~stationary.ground_truth_drift]
    results = {
        "source": "synthetic",
        "name": "12 controlled ContextEdge scenarios",
        "seed": seed,
        "created_at": time.time(),
        "settings": settings.to_dict(),
        "python": platform.python_version(),
        "artifact_directory": str(run_dir),
        "ai": classification_metrics(stationary.ground_truth_anomaly, stationary.prediction_anomaly),
        "drift": {
            "false_drift_observation_rate": float(no_drift.prediction_drift.mean()),
            "false_drift_scenario_rate": sum(s["has_false_drift"] for s in summaries[:11]) / 11,
            "detected_persistent_scenarios": sum(
                s["detection_delay_seconds"] is not None for s in summaries if s["persistent_change_expected"]
            ),
            "persistent_scenarios": sum(s["persistent_change_expected"] for s in summaries),
            "adaptation_time_simulated_seconds": adaptation_times,
        },
        "inference_latency": latency_summary(frame.inference_ms.dropna().tolist()),
        "model_sizes": {
            k: {"total_bytes": v["model_bytes"], "onnx_bytes": v["onnx_bytes"]} for k, v in metadata.items()
        },
        "onnx_parity": {k: v["onnx_parity"] for k, v in metadata.items()},
        "training_seconds": {k: v["training_seconds"] for k, v in metadata.items()},
        "scenarios": summaries,
        "limitations": [
            "Synthetic generative distributions; not real-world detection accuracy.",
            "Oracle manual context labels: context recognition accuracy is not measured here.",
            "Correlated overlapping windows violate exchangeability; alpha is an empirical target.",
            "Ground truth marks raw injected intervals; feature/sequence tails can count as false positives.",
            "Detection delay and adaptation duration use simulated sample time, not execution time.",
            "Adaptation scenario excluded from aggregate AI scores because its baseline and ground truth change; returning to the original regime is a new labelled change.",
        ],
    }
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    engine.repo.add_evaluation("synthetic", results["name"], results)
    ai = results["ai"]
    report = [
        "# ContextEdge executed synthetic evaluation",
        "",
        "These results are synthetic, not real-laptop accuracy.",
        "",
        f"Seed: {seed}. Python: {results['python']}. Five separately trained Isolation Forest + GRU/ONNX profiles.",
        "",
        f"Precision {ai['precision']:.4f}; recall {ai['recall']:.4f}; F1 {ai['f1']:.4f}; FPR {ai['false_positive_rate']:.4f}.",
        "",
        "| Scenario | Precision | Recall | F1 | Drift delay (s) | False drift observations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        report.append(
            f"| {s['scenario']} | {s['ai']['precision']:.3f} | {s['ai']['recall']:.3f} | {s['ai']['f1']:.3f} | {s['detection_delay_seconds']} | {s['false_drift_observations']} |"
        )
    report += ["", "## Limitations", ""] + ["- " + x for x in results["limitations"]]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return results
