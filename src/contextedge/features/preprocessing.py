import numpy as np
from sklearn.preprocessing import RobustScaler

from contextedge.features.extraction import FEATURE_NAMES


def validate_matrix(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
        raise ValueError("Feature matrix has invalid shape or nonfinite values")
    return matrix


def fit_scaler(values) -> RobustScaler:
    return RobustScaler(quantile_range=(10, 90)).fit(validate_matrix(values))


def sequences(values: np.ndarray, length: int, segments=None) -> np.ndarray:
    """Do not bridge calibration sessions, context switches or missing-sample gaps."""
    result = []
    for end in range(length, len(values) + 1):
        if segments is None or len(set(segments[end - length : end])) == 1:
            result.append(values[end - length : end])
    return (
        np.asarray(result, dtype=np.float32) if result else np.empty((0, length, values.shape[1]), np.float32)
    )
