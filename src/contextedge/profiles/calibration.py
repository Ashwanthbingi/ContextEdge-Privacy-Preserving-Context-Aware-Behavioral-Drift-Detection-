from __future__ import annotations

import json

import numpy as np

from contextedge.config import Settings
from contextedge.features.preprocessing import sequences, validate_matrix


def split_calibration(rows: list[dict], settings: Settings):
    if len(rows) < settings.min_calibration_samples:
        raise ValueError(
            f"Need {settings.min_calibration_samples} valid normal observations; received {len(rows)}"
        )
    values = validate_matrix(
        [json.loads(r["features"]) if isinstance(r["features"], str) else r["features"] for r in rows]
    )
    segments = [r["segment"] for r in rows]
    split = int(len(values) * 0.6)
    # Purge overlapping feature windows and temporal receptive fields at the split.
    validation_start = split + settings.feature_window + settings.sequence_length
    train, validation = values[:split], values[validation_start:]
    train_segments, validation_segments = segments[:split], segments[validation_start:]
    n_val = len(sequences(validation, settings.sequence_length, validation_segments))
    if n_val < settings.min_validation_samples:
        raise ValueError(
            "Insufficient contiguous held-out sequences; collect more uninterrupted calibration data"
        )
    if len(sequences(train, settings.sequence_length, train_segments)) < 20:
        raise ValueError("Insufficient contiguous training sequences")
    return train, validation, train_segments, validation_segments


def stability_score(values) -> float:
    array = validate_matrix(values)
    left, right = np.array_split(array, 2)
    scale = np.maximum(np.std(array, axis=0), 0.1)
    # 90th percentile tolerates a small number of noisy/unsupported dimensions.
    return float(np.quantile(np.abs(left.mean(0) - right.mean(0)) / scale, 0.9))
