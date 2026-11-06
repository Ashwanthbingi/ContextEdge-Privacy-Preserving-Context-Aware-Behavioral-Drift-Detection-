# ContextEdge

**Privacy-Preserving, Context-Aware Behavioral Drift Detection Using Edge AI**

ContextEdge learns separate local behavioural profiles for Idle, Coding, Browsing, Media, High-compute and optionally Unknown activity. It combines a calibrated Isolation Forest with a small GRU autoencoder running through ONNX Runtime, then applies ADWIN and Page-Hinkley with persistence safeguards. A Streamlit dashboard shows actual laptop observations, calibration progress, anomaly episodes, candidate adaptations and local evaluation reports.

Persistent drift means behaviour changed. It is **not a malware verdict**.

## Quick start — Windows PowerShell

Python 3.11 or 3.12 is required. The verified environment uses Python 3.11.16 and CPU-only PyTorch. Package installation requires internet; running, training and inference do not.

```powershell
cd 'E:\edge AI project'
# For a fresh machine with Python 3.11 registered:
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The existing `.venv` in this delivered workspace is already installed. To launch it:

```powershell
cd 'E:\edge AI project'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open **http://127.0.0.1:8501**. Streamlit usage analytics are disabled in `.streamlit/config.toml`. Launch from the repository root so this configuration is read.

Alternatively, use `./start.ps1`. On Linux/macOS, create a Python 3.11/3.12 virtual environment, install the same requirements and run `python -m streamlit run app.py` from the project root.

## First real monitoring session

1. Press **Start**. Resource charts populate from `psutil`; no sample/demo values are used in the dashboard.
2. Select the activity you are actually performing. Automatic context recognition is conservative; use manual labels for calibration.
3. Check **This is representative normal activity** while doing a typical, benign workload. Collect at least 240 valid observations, including long contiguous segments. At the default two-second interval this takes a little over eight minutes per context. More diverse normal sessions improve coverage.
4. Open **Behavioural profiles**, choose the context, then **Stop monitoring & train context models**. Training holds out later observations, purges overlapping windows, exports ONNX and verifies parity before activation. Insufficient sequences produce an explanation rather than a fake ready state.
5. Press **Start** again. Repeat calibration for other activities. Each context remains uncalibrated until enough of its own approved observations exist.
6. During a sustained confirmed change, inspect **Adaptation**. After at least 240 stable quarantined observations, stage a candidate and accept only a legitimate change. Previous versions remain available for rollback. Automatic acceptance is opt-in.

Calibration consent resets on application restart. Existing learned profiles, settings, events and model versions persist. An uncalibrated context displays **WARM-UP / UNCALIBRATED**, with no invented anomaly score. Windows foreground categorization is optional and disabled by default.

## Local storage and privacy

The default data directory is `%LOCALAPPDATA%\ContextEdge` on Windows or `~/.local/share/ContextEdge` elsewhere. Set `CONTEXTEDGE_DATA_DIR` to another private local folder before startup if desired:

```powershell
$env:CONTEXTEDGE_DATA_DIR = Join-Path $env:LOCALAPPDATA 'ContextEdge'
```

The dashboard and CLI must use the same data directory to share profiles. An OS file lock prevents two monitoring processes from writing the same store. Stop the dashboard monitor before running a headless monitor/benchmark against that store.

Stored data includes numerical feature vectors, context/category labels, model artifacts, decisions and necessary local audit events. The collector does not inspect keystrokes, screens, content, filenames, URLs, destinations, IPs, command lines, process paths or credentials. Executable basenames are mapped to categories locally and discarded; PID/creation-time pairs live only in memory for aggregate CPU deltas. User-configured executable mappings are retained as settings.

The default retention is 30 days for observations, calibration rows, events and evaluations. Learned model versions and adaptation audit history survive observation retention until explicitly deleted. Privacy controls stop monitoring before deleting data. See [privacy and threat model](docs/privacy.md).

## Reproduce validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/pytest-results.xml
.\.venv\Scripts\ruff.exe check src dashboard scripts tests app.py
.\.venv\Scripts\python.exe scripts/evaluate.py --output reports/synthetic --seed 42 --epochs 18
.\.venv\Scripts\python.exe scripts/benchmark.py --seconds 30 --output reports/laptop_benchmark.json
.\.venv\Scripts\python.exe scripts/benchmark_inference.py --seconds 15 --output reports/inference_benchmark.json
```

Synthetic experiments train five separate test profiles in an isolated `reports/synthetic/run-*` directory. They **never train the real monitor**. `results.json`, `decisions.csv` and `report.md` contain executed results with provenance and limitations. PyTorch weights, Isolation Forest/scaler/fusion objects and ONNX graphs remain in the corresponding run directory and are gitignored. Re-running uses a fresh isolated store.

Import an executed report into the dashboard's local evaluation section:

```powershell
.\.venv\Scripts\contextedge.exe import-evaluation reports/synthetic/results.json
```

The resource benchmark collects actual laptop telemetry. Without a calibrated active context it reports monitoring CPU/RAM and explicitly leaves inference latency unavailable. Synthetic-trained model latency is reported separately by the synthetic runner; it is measured on this CPU, but does not establish real-laptop accuracy.

`benchmark_inference.py` separately paces a trained synthetic Coding profile at four inferences per second and measures inference-only CPU/RAM/latency. It excludes collector, database and dashboard costs. Do not add or compare these measurements as if they came from one full live calibrated workload.

For manually labelled local data, use the calibration workflow and [`scripts/evaluate_labelled.py`](scripts/evaluate_labelled.py). Labels and time-aligned numerical features must be collected through an approved experiment; no accuracy is inferred from unlabelled background activity.

## CLI

```powershell
# Explicitly label representative normal coding activity, then train locally.
.\.venv\Scripts\contextedge.exe monitor --seconds 600 --context Coding --calibrate
.\.venv\Scripts\contextedge.exe train Coding
.\.venv\Scripts\contextedge.exe monitor --seconds 60 --context Coding
.\.venv\Scripts\contextedge.exe status
# Global data-directory option precedes the subcommand:
.\.venv\Scripts\contextedge.exe --data-dir 'C:\your-private-folder' benchmark --seconds 30
```

`Ctrl+C` stops headless monitoring cleanly. Training stops monitoring intentionally; click Start to resume. The dashboard's worker continues until Stop or process exit, even if its browser tab is closed.

## Implementation and evidence

- [Architecture and decision rules](docs/architecture.md)
- [Feature schema and context recognition](docs/features.md)
- [Database and artifact design](docs/database.md)
- [Evaluation methodology](docs/evaluation.md)
- [Privacy and limitations](docs/privacy.md)
- [Requirement-by-requirement acceptance checklist](docs/acceptance.md)
- [Executed synthetic report](reports/synthetic/report.md)
- [Final verification record](reports/verification.md)

Direct dependency versions are pinned in `requirements.txt`; `requirements-lock.txt` records the complete installed verification environment. ONNX exports use static batch size 1, fixed sequence length and CPUExecutionProvider. PyTorch may emit static GRU tracing warnings; correctness is checked numerically on held-out sequences before any profile activates.

## Important limitations

This is a working research application, not a validated universal detector. Synthetic distributions are deliberately controlled; the measured synthetic scores do not establish real-world precision/recall. Real normal profiles need user-labelled calibration on the target laptop. Heuristic context confidence is an evidence score, not a calibrated probability of user intent. Browsing versus media inside a browser can be ambiguous. Context switching or missing samples restarts temporal continuity conservatively.

Per-context rank thresholds are empirical: overlapping time windows are correlated, so nominal alpha is not a guaranteed long-run false-positive bound. The finite validation set limits threshold resolution. The GRU's temporal footprint deliberately delays drift confirmation to suppress spike echoes. Strict consecutive abnormal evidence can miss intermittent drift. More extensive longitudinal, independently labelled studies are needed.
