from __future__ import annotations

import time
from dataclasses import dataclass, field

import psutil

from contextedge.collector.platform.foreground import foreground_category
from contextedge.collector.platform.process_times import process_times
from contextedge.context.mappings import CATEGORIES, DEFAULT_MAPPINGS, categorize


@dataclass
class TelemetrySample:
    timestamp: float
    metrics: dict[str, float | None]
    category_counts: dict[str, int] = field(default_factory=dict)
    category_cpu: dict[str, float] = field(default_factory=dict)
    foreground: str | None = None
    elapsed: float = 0.0


class TelemetryCollector:
    """Nonblocking counter deltas. CPU priming and missing counters are explicit."""

    def __init__(self, mappings=None, foreground_enabled=False, clock=time.monotonic):
        self.mappings = dict(mappings or DEFAULT_MAPPINGS)
        self.foreground_enabled = foreground_enabled
        self.clock = clock
        self.running = False
        self._last = None
        self._disk = self._net = None
        self._process_times: dict[tuple[int, float], float] = {}

    @staticmethod
    def _safe(fn):
        try:
            return fn()
        except (OSError, NotImplementedError, AttributeError, psutil.Error):
            return None

    def start(self):
        self.running = True
        self._last = self.clock()
        self._disk = self._safe(psutil.disk_io_counters)
        self._net = self._safe(psutil.net_io_counters)
        self._process_times.clear()
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)

    def stop(self):
        self.running = False
        self._process_times.clear()

    def sample(self) -> TelemetrySample:
        if not self.running:
            raise RuntimeError("Collector is stopped")
        now = self.clock()
        elapsed = now - self._last
        memory = self._safe(psutil.virtual_memory)
        cores = self._safe(lambda: psutil.cpu_percent(interval=None, percpu=True))
        cpu = self._safe(lambda: psutil.cpu_percent(interval=None))
        disk = self._safe(psutil.disk_io_counters)
        net = self._safe(psutil.net_io_counters)
        load = self._safe(psutil.getloadavg)

        def rate(current, previous, key):
            if current is None or previous is None or elapsed <= 0:
                return None
            delta = getattr(current, key) - getattr(previous, key)
            return delta / elapsed / 1048576 if delta >= 0 else None

        metrics = {
            "cpu_percent": cpu,
            "ram_percent": memory.percent if memory else None,
            "available_gb": memory.available / 1073741824 if memory else None,
            "core_max": max(cores) if cores else None,
            "load_per_core": load[0] / (psutil.cpu_count() or 1) if load else None,
            "disk_read_mb_s": rate(disk, self._disk, "read_bytes"),
            "disk_write_mb_s": rate(disk, self._disk, "write_bytes"),
            "net_recv_mb_s": rate(net, self._net, "bytes_recv"),
            "net_sent_mb_s": rate(net, self._net, "bytes_sent"),
        }
        counts = dict.fromkeys(CATEGORIES, 0)
        category_cpu = dict.fromkeys(CATEGORIES, 0.0)
        total_cpu = total_memory = 0.0
        accessible_cpu = 0
        new_times = {}
        # No cmdline(), exe(), open_files(), connections(), environ(), or content APIs.
        for process in psutil.process_iter(attrs=["pid", "name"], ad_value=None):
            try:
                info = process.info
                category = categorize(info.get("name"), self.mappings)
                counts[category] += 1
                times = process_times(process)
                utilization = 0.0
                if times:
                    created, current = times
                    identity = (info["pid"], created)
                    accessible_cpu += 1
                    previous = self._process_times.get(identity)
                    # Windows PID 0 is System Idle Process: its time is unused
                    # CPU capacity, not application activity.
                    if info["pid"] != 0 and previous is not None and elapsed > 0:
                        utilization = max(0.0, current - previous) / elapsed * 100 / (psutil.cpu_count() or 1)
                    new_times[identity] = current
                    # Query memory only for accessible processes. On Windows the
                    # protected-process fallback repeats the same costly scan.
                    total_memory += self._safe(process.memory_percent) or 0.0
                total_cpu += utilization
                category_cpu[category] += utilization
            except (psutil.Error, OSError):
                continue
        metrics.update(
            process_count=float(sum(counts.values())),
            process_cpu=min(total_cpu, 100.0),
            process_memory=total_memory,
            process_cpu_coverage=accessible_cpu / max(1, sum(counts.values())),
        )
        self._last, self._disk, self._net = now, disk, net
        self._process_times = new_times
        fg = foreground_category(self.mappings) if self.foreground_enabled else None
        return TelemetrySample(time.time(), metrics, counts, category_cpu, fg, elapsed)
