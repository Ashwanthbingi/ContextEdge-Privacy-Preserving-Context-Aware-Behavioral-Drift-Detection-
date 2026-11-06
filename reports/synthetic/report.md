# ContextEdge executed synthetic evaluation

These results are synthetic, not real-laptop accuracy.

Seed: 42. Python: 3.11.16. Five separately trained Isolation Forest + GRU/ONNX profiles.

Precision 0.6674; recall 1.0000; F1 0.8005; FPR 0.0795.

| Scenario | Precision | Recall | F1 | Drift delay (s) | False drift observations |
|---|---:|---:|---:|---:|---:|
| normal_coding | 0.000 | 0.000 | 0.000 | None | 0 |
| normal_browsing | 0.000 | 0.000 | 0.000 | None | 0 |
| normal_media | 0.000 | 0.000 | 0.000 | None | 0 |
| normal_idle | 0.000 | 0.000 | 0.000 | None | 0 |
| normal_high_compute | 0.000 | 0.000 | 0.000 | None | 0 |
| short_cpu_spike | 0.158 | 1.000 | 0.273 | None | 0 |
| short_memory_spike | 0.188 | 1.000 | 0.316 | None | 0 |
| temporary_disk_network | 0.061 | 1.000 | 0.115 | None | 0 |
| sustained_cpu_change | 0.813 | 1.000 | 0.897 | 58.0 | 12 |
| sustained_memory_change | 0.855 | 1.000 | 0.922 | 58.0 | 7 |
| sudden_context_profile_change | 0.877 | 1.000 | 0.935 | 58.0 | 7 |
| legitimate_change_adaptation | 0.898 | 1.000 | 0.946 | 58.0 | 0 |

## Limitations

- Synthetic generative distributions; not real-world detection accuracy.
- Oracle manual context labels: context recognition accuracy is not measured here.
- Correlated overlapping windows violate exchangeability; alpha is an empirical target.
- Ground truth marks raw injected intervals; feature/sequence tails can count as false positives.
- Detection delay and adaptation duration use simulated sample time, not execution time.
- Adaptation scenario excluded from aggregate AI scores because its baseline and ground truth change; returning to the original regime is a new labelled change.
