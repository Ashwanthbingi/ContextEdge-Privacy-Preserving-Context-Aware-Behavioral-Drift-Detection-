from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_metrics(truth, prediction):
    if len(truth) != len(prediction) or not len(truth):
        raise ValueError("Ground truth and predictions must be nonempty and aligned")
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[False, True]).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "observations": len(truth),
    }


def latency_summary(values):
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    values = np.asarray(values)
    return {
        "count": len(values),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(values.max()),
    }
