# app/features.py
import pandas as pd
import numpy as np
from typing import Dict, List

# Floors to avoid divide-by-zero & crazy z when variance is tiny
SIGMA_FLOOR = {
    "temperature":   0.15,
    "humidity":      0.5,
    "wind_speed":    0.3,
    "rainfall":      0.01,
    # add a modest floor for circular z
    "wind_direction": 5.0,   # degrees
}

# Circular statistics helpers for wind direction
def circular_mean_deg(values: np.ndarray) -> float:
    """Circular mean of angles in degrees [0, 360)."""
    vals = np.deg2rad(values)
    s = np.sin(vals).mean()
    c = np.cos(vals).mean()
    mean_angle = np.rad2deg(np.arctan2(s, c))
    return mean_angle % 360

def angular_difference_deg(a: float, b: float) -> float:
    """Shortest signed angular difference a-b in degrees."""
    return (a - b + 180) % 360 - 180

# Time-of-day baseline helpers for temperature “unusual vs usual”
def minute_of_day(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    return ts.dt.hour * 60 + ts.dt.minute

def compute_tod_baseline(df: pd.DataFrame, lookback_days: int = 14) -> pd.DataFrame:
    """
    Build robust minute-of-day baselines per (station, metric).
    Expects df columns: ts, station_id, metric, value.
    Returns columns: station_id, metric, mod, median, iqr
    """
    x = df.copy()
    x["mod"] = minute_of_day(x["ts"])
    ref = (
        x.groupby(["station_id", "metric", "mod"])["value"]
         .agg(median="median",
              q25=lambda s: s.quantile(0.25),
              q75=lambda s: s.quantile(0.75))
         .reset_index()
    )
    ref["iqr"] = (ref["q75"] - ref["median"]).clip(lower=1e-6)
    return ref[["station_id", "metric", "mod", "median", "iqr"]]

def attach_tod_residuals(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """
    Attach residuals and z_tod (resid / IQR_at_minute_of_day) to df.
    """
    out = df.copy()
    out["mod"] = minute_of_day(out["ts"])
    out = out.merge(ref, on=["station_id", "metric", "mod"], how="left")
    out["resid"] = out["value"] - out["median"]
    out["z_tod"] = out["resid"] / out["iqr"].replace(0, 1e-6)
    return out

# Main feature computations
def compute_univariate_feats(df: pd.DataFrame, window_minutes: int = 90) -> pd.DataFrame:
    """
    Compute rolling features per (station, metric).
    Uses time-aware rolling windows (e.g., '90T') so gaps don’t distort stats.
    Adds robust z (MAD/IQR-like) and circular handling for wind_direction.

    Expects columns: ts (datetime-like), station_id, metric, value
    """
    if df.empty:
        return df.copy()

    x = df.copy()
    x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
    x = x.dropna(subset=["ts"])
    x = x.sort_values(["metric", "station_id", "ts"])

    win = f"{int(max(1, window_minutes))}T" # time-based window spec (minutes)

    def _grp(g: pd.DataFrame) -> pd.DataFrame:
        m = g["metric"].iloc[0]
        g = g.set_index("ts")
        v = g["value"].astype(float)

        if m == "wind_direction":
            # Circular: roll on sin/cos, derive mean angle and concentration
            rad = np.deg2rad(v)
            sin = pd.Series(np.sin(rad), index=v.index)
            cos = pd.Series(np.cos(rad), index=v.index)
            mean_sin = sin.rolling(win, min_periods=5).mean()
            mean_cos = cos.rolling(win, min_periods=5).mean()

            mu_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360
            R = np.sqrt((mean_sin ** 2) + (mean_cos ** 2)).clip(1e-9, 1.0)
            circ_std = np.rad2deg(np.sqrt(-2 * np.log(R)))  # 0..inf (deg)

            # z-like score: angular diff to rolling mean over circular std
            diff = (v - mu_angle + 180) % 360 - 180
            z = diff / circ_std.replace(0, np.nan)

            prev = v.shift(1)
            delta = (v - prev + 180) % 360 - 180
            rolling_vol = delta.abs().rolling(win, min_periods=5).mean()

            g["mu"] = mu_angle
            g["sigma"] = circ_std
            g["z"] = z
            g["z_robust"] = z # fallback identical for circular
            g["delta"] = delta
            g["rolling_vol"] = rolling_vol
            return g.reset_index()

        # Linear metrics: mean/std and robust median/MAD
        mu = v.rolling(win, min_periods=5).mean()
        sig = v.rolling(win, min_periods=5).std(ddof=0)

        med = v.rolling(win, min_periods=5).median()
        mad = (v - med).abs().rolling(win, min_periods=5).median()
        sig_rob = 1.4826 * mad

        floor = SIGMA_FLOOR.get(m, 1e-6)
        sig = sig.clip(lower=floor)
        sig_rob = sig_rob.clip(lower=floor)

        z = (v - mu) / sig
        z_robust = (v - med) / sig_rob
        delta = v.diff()
        rolling_vol = delta.abs().rolling(win, min_periods=5).mean()

        g["mu"] = mu
        g["sigma"] = sig
        g["z"] = z
        g["z_robust"] = z_robust
        g["delta"] = delta
        g["rolling_vol"] = rolling_vol
        return g.reset_index()

    return x.groupby(["metric", "station_id"], group_keys=False).apply(_grp)

# Spatial neighbor gap (sanity check / feature)
def neighbor_gap(latest: pd.DataFrame, neighbors_map: Dict[str, List]) -> pd.DataFrame:
    """
    Compute gap to neighbors for the latest snapshot per (metric, station).
    neighbors_map: {station_id: [(neighbor_id, dist_km), ...]}
    Adds 'neighbor_gap' column: linear diff (value - neighbor_median) or angular diff for wind_direction.
    """
    if latest.empty:
        return latest

    vmap = {r.station_id: r.value for r in latest.itertuples()}
    gaps = []
    for sid, metric, val in latest[["station_id", "metric", "value"]].itertuples(index=False):
        nbs = neighbors_map.get(sid, [])
        vals = [vmap[n] for n in [n for n, _ in nbs] if n in vmap]
        if metric == "wind_direction" and vals:
            ref = circular_mean_deg(np.array(vals))
            gap = angular_difference_deg(val, ref)
        else:
            ref = np.median(vals) if vals else np.nan
            gap = (val - ref) if not np.isnan(ref) else np.nan
        gaps.append(gap)

    out = latest.copy()
    out["neighbor_gap"] = gaps
    return out
