import json
import pandas as pd
from datetime import timedelta

from .config import settings
from .store import insert_alert, conn
from .features import angular_difference_deg

# Helpers
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

def _cooldown_minutes_default() -> int:
    # Global fallback
    return getattr(settings, "cooldown_minutes", 10)

# Simple per-tick delta (fallback for any metric)

def apply_simple_delta_rules(window_df: pd.DataFrame, metric: str, ts: pd.Timestamp):
    """
    Raise an ALERT when change exceeds a per-metric threshold.
    - rainfall: current 5-min value (mm) >= threshold, sustained N ticks
    - others: |Δ| >= threshold (circular for wind_direction), sustained N ticks
    Emits: type='ALERT'
    """
    thr = settings.simple_delta_threshold.get(metric, float("inf"))
    need = int(settings.simple_sustained_ticks.get(metric, 1))

    for sid, g in window_df[window_df["metric"] == metric].groupby("station_id"):
        g = g.sort_values("ts")
        if metric == "rainfall":
            vals = g["value"].tail(need)
            if len(vals) < need or vals.isna().any():
                continue
            if not (vals >= thr).all():
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
            if len(g) < need + 1:
                continue
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

            if len(deltas) < need or not all(d >= thr for d in deltas[-need:]):
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

# Rain event rules (ONSET / INTENSE / EASING / STOP)
def apply_rain_event_rules(window_df: pd.DataFrame, ts: pd.Timestamp):
    """
    window_df: rows for metric='rainfall' across stations over at least ~30–45 min.
    Uses both tick thresholds and 15-min sum (W = rain_trend_window).
    Emits: RAIN_ONSET, RAIN_INTENSE, RAIN_EASING, RAIN_STOP
    """
    W = getattr(settings, "rain_trend_window", 3)  # 3 ticks ≈ 15 min
    onset_k = getattr(settings, "rain_onset_min_intervals", 2)
    calm_mm = getattr(settings, "rain_calm_mm", 0.05)
    cd_min = getattr(settings, "rain_cooldown_minutes", _cooldown_minutes_default())

    thr_onset_tick = getattr(settings, "rain_onset_tick_mm", 0.2)
    thr_onset_S15  = getattr(settings, "rain_onset_S15_mm", 0.5)
    thr_intense_tick = getattr(settings, "rain_intense_tick_mm", 2.0)
    thr_intense_S15  = getattr(settings, "rain_intense_S15_mm", 3.0)

    for sid, g in window_df[window_df["metric"]=="rainfall"].groupby("station_id"):
        g = g.sort_values("ts").copy()
        if len(g) < max(W, onset_k) + 2:
            continue

        recent_vals = g["value"].tail(W).fillna(0)
        prev_vals   = g["value"].iloc[-2-W:-2].fillna(0) if len(g) >= W + 2 else pd.Series([], dtype=float)

        recent_sum = float(recent_vals.sum())
        prev_sum   = float(prev_vals.sum()) if not prev_vals.empty else 0.0

        # ONSET: (any of last onset_k ticks >= thr_onset_tick OR recent_sum >= thr_onset_S15) AND previously dry
        onset_tick_ok = (recent_vals.tail(onset_k) >= thr_onset_tick).any()
        onset_sum_ok  = (recent_sum >= thr_onset_S15)
        was_dry = prev_sum <= calm_mm
        if (onset_tick_ok or onset_sum_ok) and was_dry and _cooldown_ok(sid, "rainfall", "RAIN_ONSET", ts, cd_min):
            insert_alert(
                ts.isoformat(), sid, "rainfall", "RAIN_ONSET",
                severity=recent_sum,
                reason=f"onset: S15={recent_sum:.2f}mm; prev15m={prev_sum:.2f}mm",
                payload_json=json.dumps({"S15_mm": recent_sum, "prev15m_mm": prev_sum})
            )

        # INTENSE: current tick or S15 over heavy thresholds
        tick_now = float(g["value"].iloc[-1]) if pd.notna(g["value"].iloc[-1]) else 0.0
        if (tick_now >= thr_intense_tick or recent_sum >= thr_intense_S15) and _cooldown_ok(sid, "rainfall", "RAIN_INTENSE", ts, cd_min):
            insert_alert(
                ts.isoformat(), sid, "rainfall", "RAIN_INTENSE",
                severity=max(tick_now, recent_sum),
                reason=f"intense: tick={tick_now:.2f}mm; S15={recent_sum:.2f}mm",
                payload_json=json.dumps({"tick_mm": tick_now, "S15_mm": recent_sum})
            )

        # EASING: recent S15 ≤ 50% of previous S15 (configurable)
        drop_frac = getattr(settings, "rain_drop_pct_for_easing", 0.5)
        easing = (prev_sum > calm_mm) and (recent_sum <= prev_sum * (1.0 - drop_frac))
        if easing and _cooldown_ok(sid, "rainfall", "RAIN_EASING", ts, cd_min):
            drop_pct = 0.0 if prev_sum == 0 else 1.0 - (recent_sum / prev_sum)
            insert_alert(
                ts.isoformat(), sid, "rainfall", "RAIN_EASING",
                severity=drop_pct,
                reason=f"easing: S15 {recent_sum:.2f}mm ≤ {100*(1.0-drop_frac):.0f}% of prev {prev_sum:.2f}mm",
                payload_json=json.dumps({"S15_mm": recent_sum, "prev15m_mm": prev_sum, "drop_pct": drop_pct})
            )

        # STOP: last q ticks effectively dry
        q = getattr(settings, "rain_stop_quiet_intervals", 2)
        stopped = (g["value"].tail(q) <= calm_mm).all()
        if stopped and _cooldown_ok(sid, "rainfall", "RAIN_STOP", ts, cd_min):
            insert_alert(
                ts.isoformat(), sid, "rainfall", "RAIN_STOP",
                severity=0.0,
                reason=f"stopped: last {q}×5m ≤ {calm_mm}mm",
                payload_json=json.dumps({})
            )

# Wind speed rules (STRONG / VERY_STRONG)
def apply_wind_speed_rules(window_df: pd.DataFrame, ts: pd.Timestamp):
    """
    Emits WIND_STRONG (>=12 kn sustained) and WIND_VERY_STRONG (>=20 kn sustained).
    Thresholds and sustain count configurable via settings.
    """
    thr_strong = getattr(settings, "wind_strong_kn", 12.0)
    thr_very   = getattr(settings, "wind_very_strong_kn", 20.0)
    need       = int(getattr(settings, "wind_sustain_ticks", 2))
    cd_min     = _cooldown_minutes_default()

    for sid, g in window_df[window_df["metric"]=="wind_speed"].groupby("station_id"):
        g = g.sort_values("ts").tail(need)
        if len(g) < need:
            continue
        v = g["value"].astype(float)
        if v.isna().any():
            continue

        if (v >= thr_very).all() and _cooldown_ok(sid, "wind_speed", "WIND_VERY_STRONG", ts, cd_min):
            insert_alert(
                ts.isoformat(), sid, "wind_speed", "WIND_VERY_STRONG",
                severity=float(v.iloc[-1]),
                reason=f">= {thr_very} kn for {need} tick(s)",
                payload_json=json.dumps({"last_kn": float(v.iloc[-1])})
            )
        elif (v >= thr_strong).all() and _cooldown_ok(sid, "wind_speed", "WIND_STRONG", ts, cd_min):
            insert_alert(
                ts.isoformat(), sid, "wind_speed", "WIND_STRONG",
                severity=float(v.iloc[-1]),
                reason=f">= {thr_strong} kn for {need} tick(s)",
                payload_json=json.dumps({"last_kn": float(v.iloc[-1])})
            )

# Temperature vs time-of-day rules (HIGH/LOW_UNUSUAL)
def apply_temp_tod_rules(df_with_tod: pd.DataFrame, ts: pd.Timestamp,
                         z_hi: float = 3.0, z_lo: float = -3.0,
                         need: int = 2, delta_min: float = 0.2):
    """
    df_with_tod must include: ts, station_id, metric='temperature', value, resid, z_tod.
    Emits TEMP_HIGH_UNUSUAL / TEMP_LOW_UNUSUAL when z_tod beyond thresholds,
    sustained for N ticks, with small absolute movement guard.
    """
    cd_min = _cooldown_minutes_default()

    g_all = df_with_tod[df_with_tod["metric"] == "temperature"].copy()
    if g_all.empty or "z_tod" not in g_all.columns:
        return

    for sid, g in g_all.groupby("station_id"):
        g = g.sort_values("ts").tail(max(need + 1, 6))
        if g.empty or len(g) < need + 1:
            continue
        g["delta"] = g["value"].diff()

        # High
        cond_hi = (g["z_tod"] >= z_hi) & (g["delta"].abs() >= delta_min)
        if cond_hi.tail(need).all() and _cooldown_ok(sid, "temperature", "TEMP_HIGH_UNUSUAL", ts, cd_min):
            insert_alert(
                ts.isoformat(), sid, "temperature", "TEMP_HIGH_UNUSUAL",
                severity=float(g["z_tod"].iloc[-1]),
                reason=f"hotter than usual: z_tod={g['z_tod'].iloc[-1]:.2f}, resid={g['resid'].iloc[-1]:.2f}°C",
                payload_json=json.dumps({"z_tod": float(g["z_tod"].iloc[-1]), "resid": float(g["resid"].iloc[-1])})
            )

        # Low
        cond_lo = (g["z_tod"] <= z_lo) & (g["delta"].abs() >= delta_min)
        if cond_lo.tail(need).all() and _cooldown_ok(sid, "temperature", "TEMP_LOW_UNUSUAL", ts, cd_min):
            insert_alert(
                ts.isoformat(), sid, "temperature", "TEMP_LOW_UNUSUAL",
                severity=float(abs(g["z_tod"].iloc[-1])),
                reason=f"colder than usual: z_tod={g['z_tod'].iloc[-1]:.2f}, resid={g['resid'].iloc[-1]:.2f}°C",
                payload_json=json.dumps({"z_tod": float(g["z_tod"].iloc[-1]), "resid": float(g["resid"].iloc[-1])})
            )
