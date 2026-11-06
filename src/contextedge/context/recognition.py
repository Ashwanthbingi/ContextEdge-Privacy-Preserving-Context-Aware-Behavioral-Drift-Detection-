from __future__ import annotations

from dataclasses import dataclass

from contextedge.collector.telemetry import TelemetrySample
from contextedge.config import Context

CATEGORY_CONTEXT = {
    "coding": Context.CODING,
    "browsing": Context.BROWSING,
    "media": Context.MEDIA,
    "compute": Context.COMPUTE,
}


@dataclass
class ContextDecision:
    context: Context
    confidence: float
    source: str
    explanation: str


class ContextRecognizer:
    def __init__(self):
        self._candidate = None
        self._count = 0

    def recognize(self, sample: TelemetrySample, manual: str | None = None) -> ContextDecision:
        if manual:
            return ContextDecision(Context(manual), 1.0, "manual", "User-selected context")
        if sample.foreground in CATEGORY_CONTEXT:
            return ContextDecision(
                CATEGORY_CONTEXT[sample.foreground],
                0.9,
                "foreground",
                "Locally categorized foreground executable",
            )
        cpu = sample.metrics.get("cpu_percent")
        io = sum(
            sample.metrics.get(k) or 0
            for k in ("disk_read_mb_s", "disk_write_mb_s", "net_recv_mb_s", "net_sent_mb_s")
        )
        if cpu is not None and cpu < 8 and io < 0.5:
            candidate, confidence, explanation = (
                Context.IDLE,
                0.7,
                "Sustained low resource activity (heuristic)",
            )
        else:
            activity = sample.category_cpu
            total = sum(activity.values())
            top = max(activity, key=activity.get) if activity else "other"
            share = activity.get(top, 0) / max(total, 0.001)
            if top in CATEGORY_CONTEXT and share >= 0.65 and total >= 1:
                candidate, confidence = CATEGORY_CONTEXT[top], min(0.85, share)
                explanation = "Dominant aggregate application-category CPU activity"
            else:
                self._candidate, self._count = None, 0
                return ContextDecision(
                    Context.UNKNOWN, 0.0, "inferred", "Insufficient unambiguous category evidence"
                )
        self._count = self._count + 1 if candidate == self._candidate else 1
        self._candidate = candidate
        if self._count < 3:
            return ContextDecision(Context.UNKNOWN, 0.0, "inferred", "Waiting for stable context evidence")
        return ContextDecision(candidate, confidence, "inferred", explanation)
