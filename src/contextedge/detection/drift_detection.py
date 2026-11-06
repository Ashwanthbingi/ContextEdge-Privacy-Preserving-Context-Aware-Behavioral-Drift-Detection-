from __future__ import annotations

from river.drift import ADWIN, PageHinkley

from contextedge.config import Settings


class DriftDetectors:
    """Actual River ADWIN and upward Page-Hinkley on bounded anomaly evidence."""

    def __init__(self, settings: Settings, reference_scores=()):
        self.settings = settings
        self.adwin = ADWIN(delta=settings.adwin_delta, clock=1, min_window_length=5, grace_period=10)
        self.page_hinkley = PageHinkley(
            min_instances=10,
            delta=settings.page_hinkley_delta,
            threshold=settings.page_hinkley_threshold,
            alpha=0.9999,
            mode="up",
        )
        self.adwin_age = self.ph_age = settings.persistence_window + 1
        self.last_signals = {"adwin": False, "page_hinkley": False}
        for value in reference_scores:
            self.adwin.update(float(value))
            self.page_hinkley.update(float(value))
        self.adwin_age = self.ph_age = settings.persistence_window + 1

    def update(self, score: float) -> dict:
        self.adwin.update(float(score))
        self.page_hinkley.update(float(score))
        self.adwin_age = 0 if self.adwin.drift_detected else self.adwin_age + 1
        self.ph_age = 0 if self.page_hinkley.drift_detected else self.ph_age + 1
        horizon = self.settings.persistence_window
        signals = {"adwin": self.adwin_age <= horizon, "page_hinkley": self.ph_age <= horizon}
        self.last_signals = signals
        mode = self.settings.drift_confirmation
        confirmed = (
            any(signals.values())
            if mode == "either"
            else all(signals.values())
            if mode == "both"
            else signals[mode]
        )
        return {**signals, "confirmed": confirmed}

    def clear_confirmation(self):
        self.adwin_age = self.ph_age = self.settings.persistence_window + 1
