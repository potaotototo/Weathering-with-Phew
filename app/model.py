import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class IsoForestModel:
    def __init__(self, random_state=42):
        self.model = IsolationForest(n_estimators=150, contamination="auto", random_state=random_state)
        self._fitted = False

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        # Always include z, delta, rolling_vol; include neighbor_gap if present, else fill 0.
        cols = ["z", "delta", "rolling_vol"]
        Xdf = df[cols].copy()
        if "neighbor_gap" in df.columns:
            Xdf["neighbor_gap"] = df["neighbor_gap"]
        else:
            Xdf["neighbor_gap"] = 0.0
        return Xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    def fit_if_needed(self, feats: pd.DataFrame):
        if feats is None or len(feats) == 0:
            return
        X = self._build_features(feats)
        if not self._fitted and len(X) >= 200:
            self.model.fit(X)
            self._fitted = True

    def score(self, feats: pd.DataFrame) -> pd.Series:
        X = self._build_features(feats)
        if not self._fitted:
            # Cold-start fallback: shaped |z|
            return np.tanh(np.abs(feats["z"].fillna(0.0)) / 3.0)
        raw = self.model.score_samples(X)  # higher = more normal
        norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return 1.0 - norm                  # higher = more anomalous
