# app/model.py
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

_FEATS_BASE = ["delta", "rolling_vol"]
_FEATS_Z    = ["z", "z_robust"]  # we'll include whichever exist
_OPT_FEATS  = ["neighbor_gap"]   # optional extras

def _build_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = []
    # prefer both z & z_robust if present
    for c in _FEATS_Z + _FEATS_BASE + _OPT_FEATS:
        if c in df.columns:
            cols.append(c)
    Xdf = df[cols].copy()
    return Xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

def _z_eff_series(df: pd.DataFrame) -> pd.Series:
    z = df["z"] if "z" in df.columns else pd.Series(0.0, index=df.index)
    zr = df["z_robust"] if "z_robust" in df.columns else pd.Series(0.0, index=df.index)
    return np.maximum(z.abs(), zr.abs())

class IsoForestModel:
    """
    Single IsolationForest with a simple 'fit once then score' lifecycle.
    Kept for backwards compatibility.
    """
    def __init__(self, random_state=42, n_estimators=150, contamination="auto"):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._fitted = False
        self._last_fit_ts = 0.0  # epoch seconds

    def fit_if_needed(self, feats: pd.DataFrame, refit_every_sec: int = 3600, min_rows: int = 200):
        if feats is None or len(feats) < min_rows:
            return
        now = time.time()
        if (not self._fitted) or (now - self._last_fit_ts >= refit_every_sec):
            X = _build_matrix(feats)
            if len(X) >= min_rows:
                self.model.fit(X)
                self._fitted = True
                self._last_fit_ts = now

    def score(self, feats: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            # Cold-start fallback: squashed z_eff
            return np.tanh(_z_eff_series(feats) / 3.0)
        X = _build_matrix(feats)
        raw = self.model.score_samples(X)  # higher = more normal
        norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        s = 1.0 - norm  # higher = more anomalous
        return pd.Series(s, index=feats.index)

class MultiIsoForest:
    """
    Per-metric pool of IsoForestModel to avoid cross-metric contamination.
    Usage:
        self.model = MultiIsoForest()
        self.model.fit_if_needed("rainfall", feats_rain)
        scores = self.model.score("rainfall", latest_rain)
    """
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
