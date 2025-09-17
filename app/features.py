import pandas as pd
import numpy as np
from typing import Dict

SIGMA_FLOOR = {
    "temperature": 0.15,
    "humidity":    0.5,
    "wind_speed":  0.3,
    "rainfall":    0.01,
}

def _linear_feats(g, win, metric):
    g = g.sort_values("ts").set_index("ts")
    mu = g["value"].rolling(win, min_periods=5).mean()
    sig = g["value"].rolling(win, min_periods=5).std(ddof=0)
    # robust stats
    med = g["value"].rolling(win, min_periods=5).median()
    mad = (g["value"] - med).abs().rolling(win, min_periods=5).median()
    sig_rob = 1.4826 * mad

    floor = SIGMA_FLOOR.get(metric, 1e-6)
    sig      = sig.clip(lower=floor)
    sig_rob  = sig_rob.clip(lower=floor)

    g["mu"]        = mu
    g["sigma"]     = sig
    g["z"]         = (g["value"] - mu)  / sig
    g["z_robust"]  = (g["value"] - med) / sig_rob
    g["delta"]     = g["value"].diff()
    g["rolling_vol"] = g["delta"].abs().rolling(win, min_periods=5).mean()
    return g.reset_index()

def circular_mean_deg(values: np.ndarray) -> float:
    """Compute circular mean of angles in degrees (0-360)."""
    vals = np.deg2rad(values)
    s = np.sin(vals).mean()
    c = np.cos(vals).mean()
    mean_angle = np.arctan2(s, c)
    return np.rad2deg(mean_angle) % 360

def circular_std_deg(values: np.ndarray) -> float:
    """Compute circular standard deviation (degrees)."""
    vals = np.deg2rad(values)
    R = np.sqrt(np.sin(vals).mean()**2 + np.cos(vals).mean()**2)
    return np.rad2deg(np.sqrt(-2*np.log(R)))

def angular_difference_deg(a: float, b: float) -> float:
    """Shortest signed angular difference between two angles (degrees)."""
    diff = (a - b + 180) % 360 - 180
    return diff

# =============================
# Feature computation
# =============================
# expects a DataFrame with columns: ts (datetime64), station_id, metric, value

def compute_univariate_feats(df: pd.DataFrame, window_minutes: int = 90) -> pd.DataFrame:
    """Compute rolling features. Uses circular statistics for wind_direction and adds z_robust."""
    df = df.sort_values(["station_id", "ts"]).copy()
    N = max(5, int(window_minutes))

    def _grp(g: pd.DataFrame):
        m = g["metric"].iloc[0]
        if m == "wind_direction":
            # --- your existing circular logic ---
            rad = np.deg2rad(g["value"])
            sin = np.sin(rad); cos = np.cos(rad)
            mean_sin = sin.rolling(N, min_periods=5).mean()
            mean_cos = cos.rolling(N, min_periods=5).mean()
            mu_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360
            R = np.sqrt(mean_sin**2 + mean_cos**2).clip(1e-9, 1.0)
            circ_std = np.rad2deg(np.sqrt(-2*np.log(R)))
            diff = (g["value"] - mu_angle + 180) % 360 - 180
            g["mu"] = mu_angle
            g["sigma"] = circ_std
            g["z"] = diff / (g["sigma"].replace(0, np.nan))
            prev = g["value"].shift(1)
            g["delta"] = (g["value"] - prev + 180) % 360 - 180
            g["rolling_vol"] = g["delta"].abs().rolling(N, min_periods=5).mean()
            g["z_robust"] = g["z"]   # fallback for circular case
            return g
        else:
            # --- linear metrics with robust stats ---
            g["mu"]    = g["value"].rolling(N, min_periods=5).mean()
            sig        = g["value"].rolling(N, min_periods=5).std(ddof=0)

            med        = g["value"].rolling(N, min_periods=5).median()
            mad        = (g["value"] - med).abs().rolling(N, min_periods=5).median()
            sig_rob    = 1.4826 * mad

            floor = SIGMA_FLOOR.get(m, 1e-6)
            sig     = sig.clip(lower=floor)
            sig_rob = sig_rob.clip(lower=floor)

            g["sigma"]    = sig
            g["z"]        = (g["value"] - g["mu"])  / sig
            g["z_robust"] = (g["value"] - med)      / sig_rob
            g["delta"]    = g["value"].diff()
            g["rolling_vol"] = g["delta"].abs().rolling(N, min_periods=5).mean()
            return g

    return df.groupby(["metric","station_id"], group_keys=False).apply(_grp)

def neighbor_gap(latest: pd.DataFrame, neighbors_map: Dict[str, list]) -> pd.DataFrame:
    vmap = {r.station_id: r.value for r in latest.itertuples()}
    gaps = []
    for sid, metric, val in latest[["station_id","metric","value"]].itertuples(index=False):
        nbs = neighbors_map.get(sid, [])
        vals = [vmap[n] for n in [n for n, _ in nbs] if n in vmap]
        if metric == "wind_direction" and vals:
            ref = circular_mean_deg(np.array(vals))
            gap = angular_difference_deg(val, ref)
        else:
            ref = np.median(vals) if vals else np.nan
            gap = val - ref if not np.isnan(ref) else np.nan
        gaps.append(gap)
    latest = latest.copy()
    latest["neighbor_gap"] = gaps
    return latest
