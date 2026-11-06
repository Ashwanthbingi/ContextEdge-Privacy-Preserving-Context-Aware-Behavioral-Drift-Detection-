import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationDetector:
    def __init__(self, seed=42):
        self.model = IsolationForest(
            n_estimators=100, max_samples=256, contamination="auto", random_state=seed, n_jobs=1
        )
        self.ready = False

    def fit(self, normalized):
        self.model.set_params(max_samples=min(256, len(normalized)))
        self.model.fit(normalized)
        self.ready = True
        return self

    def score(self, normalized) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("Isolation Forest has not been trained")
        return -self.model.score_samples(normalized)
