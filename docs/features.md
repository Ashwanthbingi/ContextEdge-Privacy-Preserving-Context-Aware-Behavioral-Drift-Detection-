# Feature schema and context inference

Feature schema version 2 has **27 numerical features**. Only these values and necessary context/detection metadata are persisted for each observation.

| Group | Features | Units / meaning |
|---|---|---|
| CPU | cpu_percent, core_max | Overall and maximum per-core utilization, percent |
| Platform load | load_per_core | One-minute system load divided by logical cores, where supported |
| Memory | ram_percent, available_gb | Physical RAM utilization and available GiB |
| Disk | disk_read_mb_s, disk_write_mb_s | Counter deltas / measured monotonic elapsed time, MiB/s |
| Network | net_recv_mb_s, net_sent_mb_s | Aggregate byte-counter deltas, MiB/s; no endpoints |
| Processes | process_count, process_cpu, process_memory | Aggregate count, machine-normalized CPU percentage, summed process memory percentages |
| Process visibility | process_cpu_coverage | Fraction of counted processes with accessible CPU counters; protected processes are skipped |
| Categories | category_coding, category_browsing, category_media, category_compute, category_other | Fractions of counted processes in configured categories |
| Rolling CPU | cpu_mean, cpu_std, cpu_slope | Within-context mean/std and utilization points per second |
| Rolling memory | ram_slope | Utilization points per second |
| Rolling I/O | disk_mean, network_mean | Mean total throughput in the feature window |
| Availability | disk_available, network_available, load_available | Numerical flags distinguishing unsupported metrics from zero |

Default window length is eight observations. Slopes use centered least-squares time differences, not sample index. Counter resets produce a missing rate instead of a negative rate. Critical CPU/RAM absence or fewer than 65% available base metrics invalidates an observation. Missing values are represented numerically with availability flags; invalid samples reset history and are excluded from learning. A complete fresh valid window is required afterwards. Per-process access failures are skipped; platform visibility limits are documented rather than treated as malicious events.

Aggregate process CPU is computed from user+system CPU-time deltas, normalized by logical cores. PID and process creation time avoid reuse errors and remain transient in memory. Windows PID 0 (System Idle Process) is excluded from CPU activity. A minimal Windows GetProcessTimes adapter retrieves only numeric counters for queryable processes, avoiding repeated expensive psutil fallbacks for protected processes. Process memory is also limited to accessible processes; the coverage fraction makes incomplete visibility explicit. Summed process memory percentages can double-count shared pages; physical memory utilization is the primary RAM signal.

## Context evidence

1. A manual override selects the requested context with confidence 1.0 and source `manual`.
2. Optional Windows foreground lookup retrieves only HWND → PID → executable basename → configured category. No title, text, full path or content API is invoked. The stored source is `foreground`, with a heuristic evidence score 0.9.
3. Otherwise, low CPU (<8%) and low total I/O (<0.5 MiB/s), stable over three observations, suggest Idle. These thresholds classify context; they are **not normal/anomaly baselines**.
4. For other activity, a mapped category must provide at least 65% of visible aggregate process CPU and at least 1% aggregate activity, stable for three observations. Confidence is capped at 0.85.
5. Ambiguous evidence returns Unknown with confidence zero.

Supported categories map to Coding, Browsing, Media and High-compute. Users add local game/tool executable basenames in the dashboard JSON mapping editor. `python.exe` is deliberately not universally equated with coding: it may host inference, training, servers or unrelated work. Likewise, a dormant browser process alone is insufficient evidence of browsing. Browser video can remain ambiguous without a manual override. Non-Windows foreground lookup returns unavailable, preserving the aggregate/manual path.

Unknown has its own optional calibrated profile. It never silently borrows another activity's baseline. A trained context-classification model and quantitative context accuracy require separate labelled data and are not claimed here.
