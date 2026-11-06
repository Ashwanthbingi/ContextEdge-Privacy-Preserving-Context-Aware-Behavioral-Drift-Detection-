from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path


class Context(StrEnum):
    IDLE = "Idle"
    CODING = "Coding"
    BROWSING = "Browsing"
    MEDIA = "Media"
    COMPUTE = "High-compute"
    UNKNOWN = "Unknown"


def default_data_dir() -> Path:
    if os.getenv("CONTEXTEDGE_DATA_DIR"):
        return Path(os.environ["CONTEXTEDGE_DATA_DIR"]).expanduser().resolve()
    base = Path(os.getenv("LOCALAPPDATA", Path.home() / ".local" / "share"))
    return base / "ContextEdge"


@dataclass
class Settings:
    sampling_interval: float = 2.0
    feature_window: int = 8
    sequence_length: int = 12
    min_calibration_samples: int = 240
    min_validation_samples: int = 50
    false_alarm_alpha: float = 0.05
    temporal_epochs: int = 18
    temporal_hidden: int = 16
    temporal_weight: float = 0.5
    persistence_window: int = 30
    min_persistence: int = 12
    persistence_ratio: float = 0.8
    drift_confirmation: str = "either"
    adwin_delta: float = 0.002
    page_hinkley_threshold: float = 8.0
    page_hinkley_delta: float = 0.005
    context_grace: int = 4
    max_context_gap_seconds: float = 120.0
    retention_days: int = 30
    foreground_enabled: bool = False
    auto_adapt: bool = False
    adaptation_samples: int = 240
    adaptation_stability_limit: float = 0.8
    random_seed: int = 42

    def validate(self) -> Settings:
        if not 0.25 <= self.sampling_interval <= 60:
            raise ValueError("Sampling interval must be 0.25–60 seconds")
        if not 2 <= self.feature_window <= 120 or not 4 <= self.sequence_length <= 64:
            raise ValueError("Invalid feature or sequence window")
        if self.min_validation_samples < 20 or self.min_calibration_samples < max(
            80, 2 * self.min_validation_samples + 4 * self.sequence_length
        ):
            raise ValueError("Calibration must allow disjoint training, validation and sequence buffers")
        if not 0.001 <= self.false_alarm_alpha <= 0.2:
            raise ValueError("False alarm alpha must be in [0.001, 0.2]")
        if self.min_validation_samples < (1 / self.false_alarm_alpha - 1):
            raise ValueError("Validation set too small for requested conformal significance")
        if not 2 <= self.min_persistence <= self.persistence_window <= 500:
            raise ValueError("Invalid persistence window")
        if not 0.5 <= self.persistence_ratio <= 1 or not 0 <= self.temporal_weight <= 1:
            raise ValueError("Invalid ratio/weight")
        if self.drift_confirmation not in {"either", "both", "adwin", "page_hinkley"}:
            raise ValueError("Unknown drift confirmation rule")
        if not 0 < self.adwin_delta < 1 or self.page_hinkley_threshold <= 0:
            raise ValueError("Invalid drift detector threshold")
        if not 1 <= self.retention_days <= 3650 or self.context_grace < 0:
            raise ValueError("Invalid retention/grace")
        if not 1 <= self.temporal_epochs <= 200 or not 4 <= self.temporal_hidden <= 64:
            raise ValueError("Invalid temporal model size/training duration")
        if self.adaptation_samples < self.min_calibration_samples:
            raise ValueError("Adaptation needs at least the calibration sample count")
        if self.max_context_gap_seconds <= 0 or self.adaptation_stability_limit <= 0:
            raise ValueError("Invalid continuity/stability limits")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed}).validate()
