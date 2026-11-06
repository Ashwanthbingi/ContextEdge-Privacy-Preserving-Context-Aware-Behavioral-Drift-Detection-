# ContextEdge final verification

Generated 2026-11-06T09:48:18.096781+00:00 from executed test and experiment output. Python 3.11.16 on Windows.

**55 tests passed; 0 failures, 0 errors, 0 skipped.** Pytest duration: 24.07 seconds. Ruff and Python compilation checks also passed.

The suite exercises real telemetry, missing metrics, feature windows, all contexts, calibration readiness, trained scoring, both River detectors, transient/sustained distinctions, context transitions, adaptation, rollback, ONNX parity, restart, corruption, SQLite migration/errors, safe deletion, network-disabled processing, single-writer protection, labelled evaluation and Streamlit controls. Six static GRU tracing warnings were emitted; fixed-shape ONNX parity checks passed.

## Executed synthetic experiments

Five distinct normal profiles were trained: Idle, Coding, Browsing, Media, High-compute. Each used 400 normal observations, a separately fitted Isolation Forest, a GRU trained for 18 epochs, and held-out score calibration. A sixth version is the staged/accepted Coding adaptation candidate. Native PyTorch, joblib and ONNX artifacts remain in the run directory below. No synthetic profiles were installed into the real monitor.

`E:\edge AI project\reports\synthetic\run-1793957821756033300`

| Metric | Executed result |
|---|---:|
| Precision | 66.7387% |
| Recall | 100.0000% |
| F1 | 80.0518% |
| False-positive rate | 7.9545% |
| TP / FP / TN / FN | 309 / 154 / 1782 / 0 |
| False-drift observation rate | 1.3368% |
| False-drift scenario rate (includes recovery tails) | 27.2727% |
| Persistent scenarios detected | 4 / 4 |
| Maximum PyTorch/ONNX absolute error | 2.38e-07 |

All four persistent scenarios first detected drift at 58 simulated seconds after onset. All five normal scenarios and all three short-spike scenarios had zero persistent-drift classifications. Recovery tails after sustained changes account for the reported false-drift observations. Synthetic adaptation took 478 simulated seconds from detection to acceptance. The adaptation scenario is scored separately because the approved baseline changes.

## Measured edge costs

These are separate runs, not a combined calibrated live-monitor benchmark.

| Measurement | Real laptop monitor, uncalibrated | Synthetic-trained Coding inference |
|---|---:|---:|
| Duration | 30.00 s | 15.00 s |
| Observations / inferences | 14 | 60 |
| Mean RSS | 331.84 MiB | 333.89 MiB |
| Peak RSS | 332.21 MiB | 333.91 MiB |
| CPU, one-core basis | 2.083% | None |
| CPU, 16-thread machine basis | 0.130% | None |
| Mean inference latency | Not applicable | 4.952 ms |
| p95 inference latency | Not applicable | 5.646 ms |

The monitor requested two-second sampling; measured mean spacing was 1.978 seconds. Inference was paced at four calls/second. A null inference CPU result means accounting counters did not advance during short paced bursts; it is not a zero-cost claim. RSS includes Python and ML libraries. The real monitor had no trained user profile, so it deliberately produced no inference predictions.

| Synthetic context | Complete artifacts (bytes) | ONNX (bytes) |
|---|---:|---:|
| Idle | 392539 | 21418 |
| Coding | 366107 | 21418 |
| Browsing | 388178 | 21418 |
| Media | 372627 | 21418 |
| High-compute | 384120 | 21418 |

## Startup and controls

The Streamlit server starts at http://127.0.0.1:8501 and its `/_stcore/health` endpoint returns HTTP 200 / ok. Streamlit AppTest verified Start/Stop/calibration and actual telemetry display. The browser visual check was stopped by the computer-use tool because it could not verify that the active browser URL was allowed; no further UI input was attempted.

The verified demonstration store is `%TEMP%\ContextEdge-verified-runtime`, outside the source tree. The running dashboard shows its measured history and imported reports. Monitoring is stopped until Start is pressed. Normal startup without the environment override uses `%LOCALAPPDATA%\ContextEdge` instead.

```powershell
cd 'E:\edge AI project'
$env:CONTEXTEDGE_DATA_DIR = Join-Path $env:TEMP 'ContextEdge-verified-runtime'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## Remaining limitations

- Real activity profiles still require representative, explicitly labelled local calibration; no real-world precision/recall is claimed.
- Synthetic accuracy reflects one controlled seed and pronounced shifts, with oracle context labels. Normal/anomaly tails are correlated.
- Context confidence is heuristic; browser media and mixed workloads can be Unknown or need manual override.
- Protected-process counters are skipped with an explicit coverage feature; aggregate system CPU/RAM remain available.
- Conservative persistence introduces delay and can miss intermittent drift. Stability does not prove a change is legitimate.
- No encryption at rest, cross-OS real-machine run, or long-term multi-user study was completed.
- Browser visual QA was unavailable as described above.

See [the complete acceptance checklist](../docs/acceptance.md), [methodology](../docs/evaluation.md), [synthetic scenario report](synthetic/report.md), [JUnit results](pytest-results.xml), and the JSON files beside this report.
