import time
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Always use this fixed schema (order matters)
_FEAT_COLS = ["z", "z_robust", "delta", "rolling_vol", "neighbor_gap"]

def _z_eff_series(df: pd.DataFrame) -> pd.Series:
    z  = df["z"] if "z" in df.columns else pd.Series(0.0, index=df.index)
    zr = df["z_robust"] if "z_robust" in df.columns else pd.Series(0.0, index=df.index)
    return np.maximum(z.abs(), zr.abs())

def _build_matrix_fixed(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    # ensure all required columns exist
    for c in _FEAT_COLS:
        if c not in X.columns:
            X[c] = 0.0
    # minimal sanitization
    return (
        X[_FEAT_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

class IsoForestModel:
    """Single IsolationForest with fixed feature schema."""
    def __init__(self, random_state=42, n_estimators=150, contamination="auto"):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._fitted = False
        self._last_fit_ts = 0.0
        self._feat_cols = list(_FEAT_COLS)  # persisted schema

    def fit_if_needed(self, feats: pd.DataFrame, refit_every_sec: int = 3600, min_rows: int = 200):
        if feats is None or len(feats) < min_rows:
            return
        now = time.time()
        if (not self._fitted) or (now - self._last_fit_ts >= refit_every_sec):
            X = _build_matrix_fixed(feats)
            if len(X) >= min_rows:
                self.model.fit(X.values)
                self._fitted = True
                self._last_fit_ts = now

    def score(self, feats: pd.DataFrame) -> pd.Series:
        # Cold start: fall back to squashed z
        if not self._fitted:
            return np.tanh(_z_eff_series(feats) / 3.0)

        X = _build_matrix_fixed(feats)

        # Safety: if existing model was trained with a different n_features, refit once.
        try:
            n_in = getattr(self.model, "n_features_in_", X.shape[1])
            if n_in != X.shape[1]:
                # attempt quick refit with whatever feats we have now
                self.model.fit(X.values)
                self._fitted = True
                self._last_fit_ts = time.time()
        except Exception:
            # ignore and use cold-start score
            return np.tanh(_z_eff_series(feats) / 3.0)

        raw = self.model.score_samples(X.values)  # higher=more normal
        norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return pd.Series(1.0 - norm, index=feats.index)  # higher=more anomalous


class MultiIsoForest:
    """Per-metric pool so models don't mix distributions across metrics."""
    def __init__(self):
        self._models: dict[str, IsoForestModel] = {}

    def _get(self, metric: str) -> IsoForestModel:
        if metric not in self._models:
            self._models[metric] = IsoForestModel()
        return self._models[metric]

    def fit_if_needed(self, metric: str, feats: pd.DataFrame, **kw):
        self._get(metric).fit_if_needed(feats, **kw)

    def score(self, metric: str, feats: pd.DataFrame) -> pd.Series:
        return self._get(metric).score(feats)
