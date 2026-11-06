from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def empirical_pvalue(reference, score: float) -> float:
    """Smoothed upper-tail rank; finite-sample floor prevents impossible p=0."""
    reference = np.asarray(reference)
    if not len(reference):
        raise ValueError("A held-out calibration distribution is required")
    return float((1 + np.count_nonzero(reference >= score)) / (len(reference) + 1))


def evidence(reference, scores) -> np.ndarray:
    return np.asarray([-np.log(empirical_pvalue(reference, float(s))) for s in np.atleast_1d(scores)])


@dataclass
class AnomalyResult:
    score: float | None
    pvalue: float | None
    anomalous: bool
    method: str
    iforest_raw: float | None = None
    temporal_raw: float | None = None


class CalibratedFusion:
    """Train-rank transformed weighted evidence, recalibrated on untouched holdout.

    No model independence assumption: final empirical calibration includes dependence.
    Time series are not exchangeable; alpha is a target, not a formal coverage promise.
    """

    def __init__(self, if_reference, temporal_reference, validation_fused, validation_if, weight=0.5):
        self.if_reference = np.asarray(if_reference)
        self.temporal_reference = np.asarray(temporal_reference)
        self.validation_fused = np.asarray(validation_fused)
        self.validation_if = np.asarray(validation_if)
        self.weight = weight

    def fuse(self, if_score, temporal_score):
        return float(
            (1 - self.weight) * evidence(self.if_reference, if_score)[0]
            + self.weight * evidence(self.temporal_reference, temporal_score)[0]
        )

    def score(self, if_score, temporal_score, alpha) -> AnomalyResult:
        if temporal_score is None:
            p = empirical_pvalue(self.validation_if, if_score)
            method = "isolation_forest_calibrated (sequence warm-up)"
        else:
            p = empirical_pvalue(self.validation_fused, self.fuse(if_score, temporal_score))
            method = "isolation_forest + GRU/ONNX, held-out rank calibration"
        return AnomalyResult(1 - p, p, p <= alpha, method, float(if_score), temporal_score)
