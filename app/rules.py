import pandas as pd
import numpy as np
import json
from datetime import timedelta
from dateutil import parser as dtparse
from .config import settings
from .store import insert_alert, conn
from .features import angular_difference_deg 

def apply_simple_delta_rules(window_df: pd.DataFrame, metric: str, ts: pd.Timestamp):
    """
    Raise an ALERT when change exceeds a per-metric threshold.
    - For rainfall: use current 5-min value (mm) >= threshold.
    - For other metrics: use |current - previous| (circular for wind_direction).
    No WEATHER/SENSOR classification; type='ALERT'.
    """
    thr = settings.simple_delta_threshold.get(metric, float("inf"))
    need = int(settings.simple_sustained_ticks.get(metric, 1))

    for sid, g in window_df[window_df["metric"] == metric].groupby("station_id"):
        g = g.sort_values("ts")
        if metric == "rainfall":
            # Use the current 5-min mm; require N consecutive ticks >= thr
            vals = g["value"].tail(need)
            if len(vals) < need or vals.isna().any():
                continue
            ok = (vals >= thr).all()
            if not ok:
                continue
            curr = float(vals.iloc[-1])
            payload = {"value": curr, "threshold": thr, "sustained_ticks": need}
            insert_alert(
                ts.isoformat(), sid, metric, "ALERT",
                severity=curr,
                reason=f"rainfall>=threshold for {need} tick(s): {curr:.2f}mm >= {thr:.2f}mm",
                payload_json=json.dumps(payload),
            )
        else:
            # Need at least need+1 samples to compute 'need' deltas
            if len(g) < need + 1:
                continue
            # Compute per-tick deltas (circular for wind_direction)
            if metric == "wind_direction":
                curr_vals = g["value"].tail(need + 1).to_list()
                deltas = [
                    abs(angular_difference_deg(curr_vals[i+1], curr_vals[i]))
                    for i in range(len(curr_vals) - 1)
                ]
            else:
                vals = g["value"].tail(need + 1)
                if vals.isna().any():
                    continue
                deltas = (vals.diff().abs().dropna()).to_list()

            if len(deltas) < need:
                continue
            ok = all(d >= thr for d in deltas[-need:])
            if not ok:
                continue

            curr = float(g["value"].iloc[-1])
            prev = float(g["value"].iloc[-2])
            delta_last = float(deltas[-1])
            payload = {
                "prev": prev,
                "curr": curr,
                "delta_last": delta_last,
                "threshold": thr,
                "sustained_ticks": need,
            }
            insert_alert(
                ts.isoformat(), sid, metric, "ALERT",
                severity=delta_last,
                reason=f"|Δ|>={thr} for {need} tick(s); last Δ={delta_last:.2f}",
                payload_json=json.dumps(payload),
            )

def apply_rules(window_df, metric, ts):
    W = settings.sustained_minutes          # number of recent rows to check
    dmin = settings.delta_min.get(metric, 0.0)

    for sid, g in window_df.groupby("station_id"):
        g = g.sort_values("ts").tail(W)

        if len(g) < W:
            continue

        # use the stronger of classic z and robust z
        cols = [c for c in ["z","z_robust"] if c in g.columns]
        if cols:
            z_eff = g[cols].abs().max(axis=1).fillna(0.0)
        else:
            z_eff = pd.Series(0.0, index=g.index)
            
        # require both "large enough" anomaly AND some absolute movement
        cond = (z_eff > settings.z_threshold) & (g["delta"].abs() > dmin)

        if not cond.all():
            continue

        # neighbor coherence -> WEATHER/SENSOR
        nb = g["neighbor_flag"].tail(1).iloc[0] if "neighbor_flag" in g else False
        nb_count = int(g.get("neighbor_count_anom", pd.Series([0])).tail(1).iloc[0] or 0)
        is_weather = nb and (nb_count >= settings.weather_neighbor_min)

        severity = float(z_eff.tail(1).iloc[0])
        score    = float(g.get("score", pd.Series([0.0])).tail(1).iloc[0])
        ngap     = g.get("neighbor_gap", pd.Series([None])).tail(1).iloc[0]

        payload = {
            "z": float(z_eff.tail(1).iloc[0]),
            "score": score,
            "neighbor_gap": None if pd.isna(ngap) else float(ngap),
            "neighbor_count_anom": nb_count,
            "sustained_min": int(W)
        }

        insert_alert(
            ts.isoformat(),
            sid, metric,
            "WEATHER" if is_weather else "SENSOR",
            severity=severity,
            reason=f"sustained {W}m; z_eff={severity:.2f}, score={score:.2f}, Δ>|{dmin}|",
            payload_json=json.dumps(payload)
        )

# Rainfall event rules (onset / intense / easing / stop)

def _last_alert_time(station_id: str, metric: str, type_: str):
    with conn() as c:
        row = c.execute(
            "SELECT ts FROM alerts WHERE station_id=? AND metric=? AND type=? ORDER BY id DESC LIMIT 1",
            (station_id, metric, type_),
        ).fetchone()
    return pd.to_datetime(row[0]) if row else None

def _cooldown_ok(station_id: str, metric: str, type_: str, ts: pd.Timestamp, minutes: int) -> bool:
    last = _last_alert_time(station_id, metric, type_)
    if last is None:
        return True
    return (ts - last) >= pd.Timedelta(minutes=minutes)

def apply_rain_event_rules(window_df: pd.DataFrame, ts: pd.Timestamp):
    """
    window_df: rows for metric='rainfall' across stations over at least last 30–45 min.
    Requires columns: ts, station_id, value, neighbor_flag (best-effort).
    """
    W = settings.rain_trend_window
    onset_k = settings.rain_onset_min_intervals
    for sid, g in window_df.groupby("station_id"):
        g = g.sort_values("ts").copy()
        if len(g) < max(W, onset_k) + 2:
            continue

        # Recent and previous windows (5-min cadence)
        recent_vals = g["value"].tail(W).fillna(0)
        prev_vals   = g["value"].iloc[-2-W:-2].fillna(0) if len(g) >= W + 2 else pd.Series([], dtype=float)

        recent_sum = float(recent_vals.sum())
        prev_sum   = float(prev_vals.sum()) if not prev_vals.empty else 0.0

        # Latest neighbor coherence (if available)
        nb_flag = bool(g.get("neighbor_flag", pd.Series([False])).tail(1).iloc[0])

        # 1) RAIN_ONSET: enough recent rain, previously dry, neighbors coherent
        was_dry = prev_sum <= settings.rain_calm_mm
        onset_ok = (recent_vals.tail(onset_k) > settings.rain_calm_mm).sum() >= 1  # at least 1 wet tick in onset window
        if onset_ok and was_dry and nb_flag and _cooldown_ok(sid, "rainfall", "WEATHER_RAIN_ONSET", ts, settings.rain_cooldown_minutes):
            insert_alert(
                ts.isoformat(), sid, "rainfall", "WEATHER_RAIN_ONSET",
                severity=recent_sum,
                reason=f"onset: recent_sum={recent_sum:.2f}mm over {W*5}m; prev_sum={prev_sum:.2f}mm; nb={nb_flag}",
                payload_json=json.dumps({"recent_sum_mm": recent_sum, "prev_sum_mm": prev_sum})
            )

        # 2) RAIN_INTENSE: heavy recent rain
        if recent_sum >= settings.rain_intense_total_mm and nb_flag and _cooldown_ok(sid, "rainfall", "WEATHER_RAIN_INTENSE", ts, settings.rain_cooldown_minutes):
            insert_alert(
                ts.isoformat(), sid, "rainfall", "WEATHER_RAIN_INTENSE",
                severity=recent_sum,
                reason=f"intense: recent_sum={recent_sum:.2f}mm over {W*5}m; nb={nb_flag}",
                payload_json=json.dumps({"recent_sum_mm": recent_sum})
            )

        # 3) RAIN_EASING: drop vs previous window
        easing = (prev_sum > settings.rain_calm_mm) and (recent_sum <= prev_sum * (1.0 - settings.rain_drop_pct_for_easing))
        if easing and _cooldown_ok(sid, "rainfall", "WEATHER_RAIN_EASING", ts, settings.rain_cooldown_minutes):
            drop_pct = 0.0 if prev_sum == 0 else 1.0 - (recent_sum / prev_sum)
            insert_alert(
                ts.isoformat(), sid, "rainfall", "WEATHER_RAIN_EASING",
                severity=drop_pct,
                reason=f"easing: recent_sum={recent_sum:.2f}mm <= {100*settings.rain_drop_pct_for_easing:.0f}% of prev_sum={prev_sum:.2f}mm",
                payload_json=json.dumps({"recent_sum_mm": recent_sum, "prev_sum_mm": prev_sum, "drop_pct": drop_pct})
            )

        # 4) RAIN_STOP: last N intervals effectively dry
        q = settings.rain_stop_quiet_intervals
        stopped = (g["value"].tail(q) <= settings.rain_calm_mm).all()
        if stopped and _cooldown_ok(sid, "rainfall", "WEATHER_RAIN_STOP", ts, settings.rain_cooldown_minutes):
            insert_alert(
                ts.isoformat(), sid, "rainfall", "WEATHER_RAIN_STOP",
                severity=0.0,
                reason=f"stopped: last {q}×5m ≤ {settings.rain_calm_mm}mm",
                payload_json=json.dumps({})
            )
