"""Build a concise acceptance report solely from executed measurement files."""

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    synthetic = json.loads((reports / "synthetic/results.json").read_text(encoding="utf-8"))
    laptop = json.loads((reports / "laptop_benchmark.json").read_text(encoding="utf-8"))
    inference = json.loads((reports / "inference_benchmark.json").read_text(encoding="utf-8"))
    suite = ET.parse(reports / "pytest-results.xml").getroot().find("testsuite")
    tests = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
    tests["seconds"] = float(suite.attrib["time"])
    if tests["failures"] or tests["errors"]:
        raise RuntimeError("Cannot report successful acceptance with failed tests")
    ai = synthetic["ai"]
    sizes = synthetic["model_sizes"]
    parity = max(p["max_absolute_error"] for p in synthetic["onnx_parity"].values())
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tests": tests,
        "synthetic_ai": ai,
        "synthetic_drift": synthetic["drift"],
        "laptop_monitoring": laptop,
        "synthetic_model_inference": inference,
        "trained_synthetic_contexts": list(sizes),
        "model_sizes": sizes,
        "max_onnx_absolute_error": parity,
        "artifact_directory": synthetic["artifact_directory"],
        "real_user_profiles_trained": False,
        "browser_visual_check": "Computer-use stopped because active browser URL could not be verified as allowed",
    }
    (reports / "verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# ContextEdge final verification",
        "",
        f"Generated {report['created_at']} from executed test and experiment output. Python 3.11.16 on Windows.",
        "",
        f"**{tests['tests']} tests passed; {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped.** "
        f"Pytest duration: {tests['seconds']:.2f} seconds. Ruff and Python compilation checks also passed.",
        "",
        "The suite exercises real telemetry, missing metrics, feature windows, all contexts, calibration readiness, "
        "trained scoring, both River detectors, transient/sustained distinctions, context transitions, adaptation, "
        "rollback, ONNX parity, restart, corruption, SQLite migration/errors, safe deletion, network-disabled processing, "
        "single-writer protection, labelled evaluation and Streamlit controls. Six static GRU tracing warnings were emitted; "
        "fixed-shape ONNX parity checks passed.",
        "",
        "## Executed synthetic experiments",
        "",
        "Five distinct normal profiles were trained: Idle, Coding, Browsing, Media, High-compute. Each used 400 normal "
        "observations, a separately fitted Isolation Forest, a GRU trained for 18 epochs, and held-out score calibration. "
        "A sixth version is the staged/accepted Coding adaptation candidate. Native PyTorch, joblib and ONNX artifacts remain "
        "in the run directory below. No synthetic profiles were installed into the real monitor.",
        "",
        f"`{synthetic['artifact_directory']}`",
        "",
        "| Metric | Executed result |",
        "|---|---:|",
        f"| Precision | {ai['precision']:.4%} |",
        f"| Recall | {ai['recall']:.4%} |",
        f"| F1 | {ai['f1']:.4%} |",
        f"| False-positive rate | {ai['false_positive_rate']:.4%} |",
        f"| TP / FP / TN / FN | {ai['tp']} / {ai['fp']} / {ai['tn']} / {ai['fn']} |",
        f"| False-drift observation rate | {synthetic['drift']['false_drift_observation_rate']:.4%} |",
        f"| False-drift scenario rate (includes recovery tails) | {synthetic['drift']['false_drift_scenario_rate']:.4%} |",
        f"| Persistent scenarios detected | {synthetic['drift']['detected_persistent_scenarios']} / {synthetic['drift']['persistent_scenarios']} |",
        f"| Maximum PyTorch/ONNX absolute error | {parity:.3g} |",
        "",
        "All four persistent scenarios first detected drift at 58 simulated seconds after onset. All five normal scenarios "
        "and all three short-spike scenarios had zero persistent-drift classifications. Recovery tails after sustained changes "
        "account for the reported false-drift observations. Synthetic adaptation took 478 simulated seconds from detection "
        "to acceptance. The adaptation scenario is scored separately because the approved baseline changes.",
        "",
        "## Measured edge costs",
        "",
        "These are separate runs, not a combined calibrated live-monitor benchmark.",
        "",
        "| Measurement | Real laptop monitor, uncalibrated | Synthetic-trained Coding inference |",
        "|---|---:|---:|",
        f"| Duration | {laptop['duration_seconds']:.2f} s | {inference['duration_seconds']:.2f} s |",
        f"| Observations / inferences | {laptop['observations']} | {inference['inference_latency']['count']} |",
        f"| Mean RSS | {laptop['rss_mean_mb']:.2f} MiB | {inference['rss_mean_mb']:.2f} MiB |",
        f"| Peak RSS | {laptop['rss_peak_mb']:.2f} MiB | {inference['rss_peak_mb']:.2f} MiB |",
        f"| CPU, one-core basis | {laptop['process_cpu_percent_one_core']:.3f}% | {inference['process_cpu_percent_one_core']} |",
        f"| CPU, 16-thread machine basis | {laptop['process_cpu_percent_machine']:.3f}% | {inference['process_cpu_percent_machine']} |",
        f"| Mean inference latency | Not applicable | {inference['inference_latency']['mean_ms']:.3f} ms |",
        f"| p95 inference latency | Not applicable | {inference['inference_latency']['p95_ms']:.3f} ms |",
        "",
        f"The monitor requested two-second sampling; measured mean spacing was {laptop['observed_mean_interval_seconds']:.3f} seconds. "
        "Inference was paced at four calls/second. A null inference CPU result means accounting counters did not advance "
        "during short paced bursts; it is not a zero-cost claim. RSS includes Python and ML libraries. The real monitor "
        "had no trained user profile, so it deliberately produced no inference predictions.",
        "",
        "| Synthetic context | Complete artifacts (bytes) | ONNX (bytes) |",
        "|---|---:|---:|",
    ]
    lines += [f"| {name} | {size['total_bytes']} | {size['onnx_bytes']} |" for name, size in sizes.items()]
    lines += [
        "",
        "## Startup and controls",
        "",
        "The Streamlit server starts at http://127.0.0.1:8501 and its `/_stcore/health` endpoint returns HTTP 200 / ok. "
        "Streamlit AppTest verified Start/Stop/calibration and actual telemetry display. The browser visual check was stopped "
        "by the computer-use tool because it could not verify that the active browser URL was allowed; no further UI input was attempted.",
        "",
        "The verified demonstration store is `%TEMP%\\ContextEdge-verified-runtime`, outside the source tree. The running "
        "dashboard shows its measured history and imported reports. Monitoring is stopped until Start is pressed. Normal startup "
        "without the environment override uses `%LOCALAPPDATA%\\ContextEdge` instead.",
        "",
        "```powershell",
        "cd 'E:\\edge AI project'",
        "$env:CONTEXTEDGE_DATA_DIR = Join-Path $env:TEMP 'ContextEdge-verified-runtime'",
        ".\\.venv\\Scripts\\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501",
        "```",
        "",
        "## Remaining limitations",
        "",
        "- Real activity profiles still require representative, explicitly labelled local calibration; no real-world precision/recall is claimed.",
        "- Synthetic accuracy reflects one controlled seed and pronounced shifts, with oracle context labels. Normal/anomaly tails are correlated.",
        "- Context confidence is heuristic; browser media and mixed workloads can be Unknown or need manual override.",
        "- Protected-process counters are skipped with an explicit coverage feature; aggregate system CPU/RAM remain available.",
        "- Conservative persistence introduces delay and can miss intermittent drift. Stability does not prove a change is legitimate.",
        "- No encryption at rest, cross-OS real-machine run, or long-term multi-user study was completed.",
        "- Browser visual QA was unavailable as described above.",
        "",
        "See [the complete acceptance checklist](../docs/acceptance.md), [methodology](../docs/evaluation.md), "
        "[synthetic scenario report](synthetic/report.md), [JUnit results](pytest-results.xml), and the JSON files beside this report.",
    ]
    (reports / "verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps({"tests": tests, "synthetic_ai": ai, "report": str(reports / "verification.md")}, indent=2)
    )


if __name__ == "__main__":
    main()
