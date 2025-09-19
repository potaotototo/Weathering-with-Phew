import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from .config import settings
from .store import insert_alert, conn
from .features import angular_difference_deg

# Cache for baseline
_TEMP_BASELINE_DF: pd.DataFrame | None = None
_TEMP_BASELINE_BUILT_AT: datetime | None = None

# Tuning
_TEMP_LOOKBACK_DAYS = 21 # use last ~3 weeks for baseline
_TEMP_MIN_DAYS = 7 # require >=7 distinct days per (station, minute-of-day)
_TEMP_SIGMA_FLOOR = 0.4 # °C floor for robust sigma
_TEMP_REBUILD_EVERY_MIN = 180 # rebuild every 3h

# Alert thresholds (human-intuitive)
_TEMP_HIGH_RESID = 2.5 # °C hotter than usual
_TEMP_LOW_RESID  = 3.0 # °C colder than usual
_TEMP_SUSTAINED_TICKS = 2 # require N consecutive ticks

# cool-down for simple delta alerts (minutes)
_SIMPLE_DELTA_COOLDOWN_MIN = getattr(settings, "simple_delta_cooldown_minutes", 3)

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
    thr  = settings.simple_delta_threshold.get(metric, float("inf"))
    need = int(settings.simple_sustained_ticks.get(metric, 1))
    cd_min = _SIMPLE_DELTA_COOLDOWN_MIN

    for sid, g in window_df[window_df["metric"] == metric].groupby("station_id"):
        g = g.sort_values("ts")

        if metric == "rainfall":
            vals = g["value"].tail(need)
            if len(vals) < need or vals.isna().any():
                continue
            ok_now = (vals >= thr).all()
            # rising-edge: previously not all ticks >= thr
            prev_vals = g["value"].iloc[-(need+1):-1] if len(g) >= need+1 else pd.Series([], dtype=float)
            ok_prev = (len(prev_vals) == need) and (prev_vals >= thr).all()
            if not ok_now or ok_prev:
                continue
            # cooldown to suppress re-fires while latest ts is unchanged
            if not _cooldown_ok(sid, metric, "ALERT", ts, cd_min):
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

            # per-tick deltas
            if metric == "wind_direction":
                vals = g["value"].tail(need + 1).to_list()
                deltas = [
                    abs(angular_difference_deg(vals[i+1], vals[i]))
                    for i in range(len(vals) - 1)
                ]
            else:
                vals = g["value"].tail(need + 1)
                if vals.isna().any():
                    continue
                deltas = (vals.diff().abs().dropna()).to_list()

            if len(deltas) < need:
                continue

            ok_now  = all(d >= thr for d in deltas[-need:])
            ok_prev = (len(deltas) >= need + 1) and all(d >= thr for d in deltas[-(need+1):-1])
            if not ok_now or ok_prev:
                continue

            # cooldown to avoid duplicates while same 5-min ts is being processed by multiple loops
            if not _cooldown_ok(sid, metric, "ALERT", ts, cd_min):
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
    Emits: RAIN_ONSET, RAIN_INTENSE, RAIN_EASING, RAIN_STOP.
    """
    W = getattr(settings, "rain_trend_window", 3) # 3 ticks ≈ 15 min
    onset_k = getattr(settings, "rain_onset_min_intervals", 2)
    calm_mm = getattr(settings, "rain_calm_mm", 0.05)
    cd_min  = getattr(settings, "rain_cooldown_minutes", 20)

    thr_onset_tick   = getattr(settings, "rain_onset_tick_mm", 0.2)
    thr_onset_S15    = getattr(settings, "rain_onset_S15_mm", 0.5)
    thr_intense_tick = getattr(settings, "rain_intense_tick_mm", 2.0)
    thr_intense_S15  = getattr(settings, "rain_intense_S15_mm", 3.0)

    q = getattr(settings, "rain_stop_quiet_intervals", 2) # STOP quiet tail
    prev_k = getattr(settings, "rain_stop_prev_window", 6) # need prior wet within this many ticks (≈30m)

    df = window_df[window_df["metric"] == "rainfall"].copy()
    if df.empty:
        return

    for sid, g in df.groupby("station_id"):
        g = g.sort_values("ts").copy()
        if len(g) < max(W, onset_k, q + prev_k) + 2:
            continue

        # rolling window
        recent_vals = g["value"].tail(W).fillna(0) # last ~15m
        prev_vals   = g["value"].iloc[-2-W:-2].fillna(0) if len(g) >= W+2 else pd.Series([], dtype=float)

        recent_sum = float(recent_vals.sum())
        prev_sum   = float(prev_vals.sum()) if not prev_vals.empty else 0.0

        # RAIN_ONSET
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

        # RAIN_INTENSE 
        tick_now = float(g["value"].iloc[-1]) if pd.notna(g["value"].iloc[-1]) else 0.0
        if (tick_now >= thr_intense_tick or recent_sum >= thr_intense_S15) and _cooldown_ok(sid, "rainfall", "RAIN_INTENSE", ts, cd_min):
            insert_alert(
                ts.isoformat(), sid, "rainfall", "RAIN_INTENSE",
                severity=max(tick_now, recent_sum),
                reason=f"intense: tick={tick_now:.2f}mm; S15={recent_sum:.2f}mm",
                payload_json=json.dumps({"tick_mm": tick_now, "S15_mm": recent_sum})
            )

        # RAIN_EASING 
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

        # RAIN_STOP 
        # last q ticks effectively dry, and there was rain shortly before
        q = getattr(settings, "rain_stop_quiet_intervals", 2)
        prev_k = getattr(settings, "rain_stop_prev_window", 6) # ~30m lookback at 5m cadence
        calm_mm = getattr(settings, "rain_calm_mm", 0.05)

        vals = pd.to_numeric(g["value"], errors="coerce").fillna(0.0)

        # Dry in the most recent q ticks
        stopped_now = (vals.tail(q) <= calm_mm).all()

        # Detect wet tick in the window just BEFORE the dry streak
        if len(vals) >= q + prev_k:
            prev_span = vals.iloc[-(q + prev_k):-q] # the [q ... q + prev_k] window
        else:
            prev_span = vals.iloc[0:0]
        previously_wet = (prev_span > calm_mm).any()

        if stopped_now and previously_wet and _cooldown_ok(sid, "rainfall", "RAIN_STOP", ts, cd_min):
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

def _build_temp_baseline() -> pd.DataFrame | None:
    """
    Build per-station, per-minute-of-day (SGT) baseline:
      mu = median(value), sigma = 1.4826 * MAD(value)
    Enforces sigma floor and minimum day coverage.
    """
    cutoff_sql = f"datetime('now','-{_TEMP_LOOKBACK_DAYS} days')"

    with conn() as c:
        df = pd.read_sql_query(
            f"""
            SELECT ts, station_id, value
            FROM readings
            WHERE metric='temperature' AND ts >= {cutoff_sql}
            """,
            c,
        )

    if df.empty:
        return None

    ts_utc = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    ts_sgt = ts_utc.dt.tz_convert("Asia/Singapore")
    df["mod"] = ts_sgt.dt.hour * 60 + ts_sgt.dt.minute # minute-of-day (SGT)
    df["date_sgt"] = ts_sgt.dt.date

    def _agg(g: pd.DataFrame):
        vals = g["value"].astype(float).dropna()
        if vals.empty:
            return pd.Series({"mu": np.nan, "sigma": np.nan, "days": 0})
        mu = float(np.median(vals))
        mad = float(np.median(np.abs(vals - mu)))
        sigma = 1.4826 * mad
        days = int(g["date_sgt"].nunique())
        return pd.Series({"mu": mu, "sigma": sigma, "days": days})

    base = (
    df.groupby(["station_id", "mod"])
      .apply(_agg)
      .reset_index()
    )

    # floor sigma & filter on minimum day coverage
    base["sigma"] = base["sigma"].clip(lower=_TEMP_SIGMA_FLOOR)
    base.loc[base["days"] < _TEMP_MIN_DAYS, ["mu", "sigma"]] = np.nan

    return base[["station_id", "mod", "mu", "sigma", "days"]].reset_index(drop=True)

def ensure_temp_baseline(force: bool = False) -> bool:
    """Rebuild if missing or stale; returns True if available."""
    global _TEMP_BASELINE_DF, _TEMP_BASELINE_BUILT_AT
    now = datetime.now(timezone.utc)
    if (not force) and _TEMP_BASELINE_DF is not None and _TEMP_BASELINE_BUILT_AT is not None:
        age_min = (now - _TEMP_BASELINE_BUILT_AT).total_seconds() / 60.0
        if age_min < _TEMP_REBUILD_EVERY_MIN:
            return True

    base = _build_temp_baseline()
    if base is None or base.empty:
        _TEMP_BASELINE_DF = None
        _TEMP_BASELINE_BUILT_AT = None
        return False

    _TEMP_BASELINE_DF = base
    _TEMP_BASELINE_BUILT_AT = now
    return True

def apply_temperature_baseline_rules(window_df: pd.DataFrame, ts: pd.Timestamp):
    """
    Fire TEMP_HIGH_UNUSUAL / TEMP_LOW_UNUSUAL when residual vs ToD baseline
    is sustained for N ticks. Severity is |resid| in °C. z_tod in payload.

    window_df must have columns: ts, station_id, metric, value (only temperature rows will be used).
    """
    if window_df is None or window_df.empty:
        return
    if not ensure_temp_baseline():
        return

    df = window_df.copy()
    df = df[df["metric"] == "temperature"]
    if df.empty:
        return

    # compute minute-of-day (SGT) for join
    ts_utc = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    ts_sgt = ts_utc.dt.tz_convert("Asia/Singapore")
    df["mod"] = ts_sgt.dt.hour * 60 + ts_sgt.dt.minute

    base = _TEMP_BASELINE_DF  # cached df
    merged = df.merge(base, on=["station_id", "mod"], how="left")

    # residual & z_tod
    merged["resid"] = merged["value"] - merged["mu"]
    merged["z_tod"] = merged["resid"] / merged["sigma"]

    need = _TEMP_SUSTAINED_TICKS

    for sid, g in merged.groupby("station_id"):
        g = g.sort_values("ts")
        if len(g) < need or g["resid"].tail(need).isna().any():
            continue
        last_ts = pd.to_datetime(g["ts"].iloc[-1], utc=True, errors="coerce")

        last_resids = g["resid"].tail(need).to_numpy()

        # HIGH
        if np.all(last_resids >= _TEMP_HIGH_RESID):
            last = g.tail(1).iloc[0]
            payload = {
                "mu_tod": float(last["mu"]) if pd.notna(last["mu"]) else None,
                "sigma_tod": float(last["sigma"]) if pd.notna(last["sigma"]) else None,
                "resid": float(last["resid"]) if pd.notna(last["resid"]) else None,
                "z_tod": float(last["z_tod"]) if pd.notna(last["z_tod"]) else None,
                "sustained_ticks": int(need),
            }
            insert_alert(
                ts.isoformat(),
                sid,
                "temperature",
                "TEMP_HIGH_UNUSUAL",
                severity=abs(float(last["resid"])),
                reason=f"hotter than usual: z_tod={payload['z_tod']:.2f}, resid={payload['resid']:.2f}°C",
                payload_json=json.dumps(payload),
            )
            continue

        # LOW
        if np.all(last_resids <= -_TEMP_LOW_RESID):
            last = g.tail(1).iloc[0]
            payload = {
                "mu_tod": float(last["mu"]) if pd.notna(last["mu"]) else None,
                "sigma_tod": float(last["sigma"]) if pd.notna(last["sigma"]) else None,
                "resid": float(last["resid"]) if pd.notna(last["resid"]) else None,
                "z_tod": float(last["z_tod"]) if pd.notna(last["z_tod"]) else None,
                "sustained_ticks": int(need),
            }
            insert_alert(
                ts.isoformat(),
                sid,
                "temperature",
                "TEMP_LOW_UNUSUAL",
                severity=abs(float(last["resid"])),
                reason=f"colder than usual: z_tod={payload['z_tod']:.2f}, resid={payload['resid']:.2f}°C",
                payload_json=json.dumps(payload),
            )
