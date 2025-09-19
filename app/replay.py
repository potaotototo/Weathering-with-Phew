import re
import sys
import json
import numpy as np
import pandas as pd
from datetime import timezone
from dateutil import parser as dtparse

from .config import settings
from .store import conn, get_readings_between, delete_alerts_range, insert_alert
from .features import angular_difference_deg
from .rules import ensure_temp_baseline # uses your cached baseline builder

# normalize any human date/time to UTC 'YYYY-MM-DD HH:MM:SS'
def _to_utc_iso(s: str) -> str:
    """
    Accepts strings like:
      - '2025/09/16'
      - '2025-09-16 18:00'
      - '2025-09-16T18:00+08:00'
    If no timezone is present, interpret as Asia/Singapore.
    """
    dt = dtparse.parse(s)
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        # treat naive inputs as Singapore time
        ts = ts.tz_localize("Asia/Singapore")
    ts_utc = ts.tz_convert("UTC")
    return ts_utc.strftime("%Y-%m-%d %H:%M:%S")


def _is_date_only(s: str) -> bool:
    return bool(re.fullmatch(r"\s*\d{4}[-/]\d{2}[-/]\d{2}\s*", s))


def parse_human_range(*args) -> tuple[str, str]:
    """
    Accept:
      - two args: since until
      - one arg:  'YYYY/MM/DD to YYYY/MM/DD'
      - one arg:  'YYYY-MM-DD'   (interpreted as whole day in SGT)
    Returns (since_utc, until_utc) as strings.
    """
    if len(args) == 1:
        s = (args[0] or "").strip()
        # single arg 'A to B'
        m = re.search(r"\bto\b", s, flags=re.IGNORECASE)
        if m:
            a = s[:m.start()].strip()
            b = s[m.end():].strip()
            # inclusive date-only end → add 1 day
            if _is_date_only(b):
                b_end = (pd.Timestamp(dtparse.parse(b))
                           .tz_localize("Asia/Singapore") + pd.Timedelta(days=1))
                return _to_utc_iso(a), b_end.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
            return _to_utc_iso(a), _to_utc_iso(b)

        # single date → whole day
        d0 = pd.Timestamp(dtparse.parse(s)).tz_localize("Asia/Singapore")
        d1 = d0 + pd.Timedelta(days=1)
        return (d0.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S"),
                d1.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S"))

    if len(args) >= 2:
        a0 = (args[0] or "").strip()
        a1 = (args[1] or "").strip()
        # if first arg is actually 'A to B' as a whole string
        if re.search(r"\bto\b", a0, re.IGNORECASE):
            return parse_human_range(a0)

        if _is_date_only(a1):
            # inclusive end date (whole day)
            d1 = (pd.Timestamp(dtparse.parse(a1))
                    .tz_localize("Asia/Singapore") + pd.Timedelta(days=1))
            return _to_utc_iso(a0), d1.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
        return _to_utc_iso(a0), _to_utc_iso(a1)

    raise ValueError("Provide a date range, e.g. '2025/09/17 to 2025/09/18'")

def _load_df(metric: str, since_utc: str, until_utc: str) -> pd.DataFrame:
    rows = get_readings_between(metric, since_utc, until_utc)
    if not rows:
        return pd.DataFrame(columns=["ts","station_id","metric","value"])
    df = pd.DataFrame(rows, columns=["ts","station_id","metric","value"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.sort_values(["station_id","ts"])

def replay_rainfall(since_utc: str, until_utc: str) -> int:
    """Recreate RAIN_* alerts using a vectorized version of your rules."""
    df = _load_df("rainfall", since_utc, until_utc)
    if df.empty:
        return 0

    W = getattr(settings, "rain_trend_window", 3) 
    calm = getattr(settings, "rain_calm_mm", 0.05)
    onset_k = getattr(settings, "rain_onset_min_intervals", 2)
    thr_onset_tick = getattr(settings, "rain_onset_tick_mm", 0.2)
    thr_onset_S15  = getattr(settings, "rain_onset_S15_mm", 0.5)
    thr_int_tick   = getattr(settings, "rain_intense_tick_mm", 2.0)
    thr_int_S15    = getattr(settings, "rain_intense_S15_mm", 3.0)
    stop_q         = getattr(settings, "rain_stop_quiet_intervals", 2)

    inserted = 0
    for sid, g in df.groupby("station_id"):
        v = g["value"].fillna(0.0).astype(float).to_numpy()
        ts = g["ts"].to_numpy()

        # rolling sums over W
        s15 = pd.Series(v).rolling(W, min_periods=W).sum().to_numpy()
        s15_prev = pd.Series(v).shift(2).rolling(W, min_periods=W).sum().to_numpy()

        # ONSET (edge)
        cond_onset = ((pd.Series(v).rolling(onset_k, min_periods=1).max().to_numpy() >= thr_onset_tick) | (s15 >= thr_onset_S15)) & (np.nan_to_num(s15_prev, nan=0.0) <= calm)
        onset_edge = cond_onset & np.logical_not(np.roll(cond_onset, 1))
        onset_edge[0] = cond_onset[0]
        idx = np.where(onset_edge)[0]
        for i in idx:
            payload = {"S15_mm": float(s15[i]) if not np.isnan(s15[i]) else 0.0,
                       "prev15m_mm": float(s15_prev[i]) if not np.isnan(s15_prev[i]) else 0.0}
            insert_alert(ts[i].isoformat(), sid, "rainfall", "RAIN_ONSET", severity=float(payload["S15_mm"]),
                         reason=f"onset: S15={payload['S15_mm']:.2f}mm; prev15m={payload['prev15m_mm']:.2f}mm",
                         payload_json=json.dumps(payload))
            inserted += 1

        # INTENSE (edge)
        cond_int = (v >= thr_int_tick) | (s15 >= thr_int_S15)
        int_edge = cond_int & np.logical_not(np.roll(cond_int, 1))
        int_edge[0] = cond_int[0]
        idx = np.where(int_edge)[0]
        for i in idx:
            payload = {"tick_mm": float(v[i]), "S15_mm": float(s15[i]) if not np.isnan(s15[i]) else 0.0}
            insert_alert(ts[i].isoformat(), sid, "rainfall", "RAIN_INTENSE", severity=max(payload["tick_mm"], payload["S15_mm"]),
                         reason=f"intense: tick={payload['tick_mm']:.2f}mm; S15={payload['S15_mm']:.2f}mm",
                         payload_json=json.dumps(payload))
            inserted += 1

        # EASING (edge)
        drop_frac = getattr(settings, "rain_drop_pct_for_easing", 0.5)
        easing = (np.nan_to_num(s15_prev, nan=0.0) > calm) & (s15 <= (1.0 - drop_frac) * np.nan_to_num(s15_prev, nan=0.0))
        easing_edge = easing & np.logical_not(np.roll(easing, 1))
        easing_edge[0] = easing[0]
        idx = np.where(easing_edge)[0]
        for i in idx:
            prev = float(s15_prev[i]) if not np.isnan(s15_prev[i]) else 0.0
            cur  = float(s15[i]) if not np.isnan(s15[i]) else 0.0
            payload = {"S15_mm": cur, "prev15m_mm": prev, "drop_pct": 0.0 if prev == 0 else 1.0 - (cur/prev)}
            insert_alert(ts[i].isoformat(), sid, "rainfall", "RAIN_EASING", severity=payload["drop_pct"],
                         reason=f"easing: S15 {cur:.2f}mm ≤ {(1.0-drop_frac)*100:.0f}% of prev {prev:.2f}mm",
                         payload_json=json.dumps(payload))
            inserted += 1

        # STOP (edge)
        wet = v > calm
        dry_q = ~(pd.Series(wet).rolling(stop_q, min_periods=stop_q).max().fillna(1).astype(bool).to_numpy())
        stop_edge = dry_q & np.logical_not(np.roll(dry_q, 1))
        stop_edge[0] = dry_q[0]
        idx = np.where(stop_edge)[0]
        for i in idx:
            insert_alert(ts[i].isoformat(), sid, "rainfall", "RAIN_STOP", severity=0.0,
                         reason=f"stopped: last {stop_q}×5m ≤ {calm}mm", payload_json=json.dumps({}))
            inserted += 1

    return inserted

def replay_simple_delta(metric: str, since_utc: str, until_utc: str) -> int:
    thr  = settings.simple_delta_threshold.get(metric, float("inf"))
    need = int(settings.simple_sustained_ticks.get(metric, 1))

    df = _load_df(metric, since_utc, until_utc)
    if df.empty:
        return 0

    inserted = 0
    for sid, g in df.groupby("station_id"):
        v  = g["value"].astype(float).to_numpy()
        ts = g["ts"].to_numpy()
        if len(v) < need + 1:
            continue
        deltas = np.abs(np.diff(v))
        ok_now  = pd.Series(deltas).rolling(need, min_periods=need).min().fillna(-1).to_numpy() >= thr
        ok_prev = np.roll(ok_now, 1)  # was already satisfied one step ago?
        edge = ok_now & ~ok_prev
        idx = np.where(edge)[0]  # index refers to delta index ⇒ alert at ts[i+1]
        for i in idx:
            insert_alert(ts[i+1].isoformat(), sid, metric, "ALERT",
                         severity=float(deltas[i]),
                         reason=f"|Δ|>={thr} for {need} tick(s); last Δ={deltas[i]:.2f}",
                         payload_json=json.dumps({"delta_last": float(deltas[i]), "threshold": thr, "sustained_ticks": need}))
            inserted += 1
    return inserted

def replay_wind_direction(since_utc: str, until_utc: str) -> int:
    metric = "wind_direction"
    thr  = settings.simple_delta_threshold.get(metric, 35.0)
    need = int(settings.simple_sustained_ticks.get(metric, 2))

    df = _load_df(metric, since_utc, until_utc)
    if df.empty:
        return 0

    inserted = 0
    for sid, g in df.groupby("station_id"):
        vals = g["value"].astype(float).to_list()
        ts   = g["ts"].to_list()
        if len(vals) < need + 1:
            continue
        deltas = [abs(angular_difference_deg(vals[i+1], vals[i])) for i in range(len(vals)-1)]
        s = pd.Series(deltas)
        ok_now  = (s.rolling(need, min_periods=need).min() >= thr).to_numpy()
        ok_prev = np.roll(ok_now, 1)
        edge = ok_now & ~ok_prev
        idx = np.where(edge)[0]
        for i in idx:
            insert_alert(ts[i+1].isoformat(), sid, metric, "ALERT",
                         severity=float(deltas[i]),
                         reason=f"|Δ|>={thr} for {need} tick(s); last Δ={deltas[i]:.2f}",
                         payload_json=json.dumps({"delta_last": float(deltas[i]), "threshold": thr, "sustained_ticks": need}))
            inserted += 1
    return inserted

def replay_wind_speed(since_utc: str, until_utc: str) -> int:
    metric = "wind_speed"
    thr_strong = getattr(settings, "wind_strong_kn", 12.0)
    thr_very   = getattr(settings, "wind_very_strong_kn", 20.0)
    need       = int(getattr(settings, "wind_sustain_ticks", 2))

    df = _load_df(metric, since_utc, until_utc)
    if df.empty: 
        return 0

    inserted = 0
    for sid, g in df.groupby("station_id"):
        v = g["value"].astype(float).to_numpy()
        ts = g["ts"].to_numpy()
        if len(v) < need:
            continue
        ge = pd.Series(v)
        ok_vstrong = (ge >= thr_very).rolling(need, min_periods=need).min().fillna(0).astype(bool).to_numpy()
        ok_strong  = (ge >= thr_strong).rolling(need, min_periods=need).min().fillna(0).astype(bool).to_numpy()

        edge_vs = ok_vstrong & ~np.roll(ok_vstrong, 1)
        edge_vs[0] = ok_vstrong[0]
        idx = np.where(edge_vs)[0]
        for i in idx:
            insert_alert(ts[i].isoformat(), sid, metric, "WIND_VERY_STRONG",
                         severity=float(v[i]),
                         reason=f">= {thr_very} kn for {need} tick(s)",
                         payload_json=json.dumps({"last_kn": float(v[i])}))
            inserted += 1

        # only strong edges that are not already “very strong”
        ok_strong_only = ok_strong & ~ok_vstrong
        edge_s = ok_strong_only & ~np.roll(ok_strong_only, 1)
        edge_s[0] = ok_strong_only[0]
        idx = np.where(edge_s)[0]
        for i in idx:
            insert_alert(ts[i].isoformat(), sid, metric, "WIND_STRONG",
                         severity=float(v[i]),
                         reason=f">= {thr_strong} kn for {need} tick(s)",
                         payload_json=json.dumps({"last_kn": float(v[i])}))
            inserted += 1
    return inserted

def replay(metrics: list[str], since: str, until: str | None = None, delete_existing: bool = True) -> int:
    since_utc, until_utc = parse_human_range(since) if until is None else parse_human_range(since, until)
    total = 0
    for m in metrics:
        if delete_existing:
            delete_alerts_range(metric=m if m!="all" else None, since_utc=since_utc, until_utc=until_utc)
        if m == "rain" or m == "rainfall":
            total += replay_rainfall(since_utc, until_utc)
        elif m in ("wind_dir","wind_direction"):
            total += replay_wind_direction(since_utc, until_utc)
        elif m == "wind_speed":
            total += replay_wind_speed(since_utc, until_utc)
        elif m in ("humidity","temperature"):
            total += replay_simple_delta(m, since_utc, until_utc)
        elif m == "all":
            total += replay_rainfall(since_utc, until_utc)
            total += replay_wind_direction(since_utc, until_utc)
            total += replay_wind_speed(since_utc, until_utc)
            total += replay_simple_delta("humidity", since_utc, until_utc)
            total += replay_simple_delta("temperature", since_utc, until_utc)
        pass
    return total

if __name__ == "__main__":
    # Examples:
    #   python -m app.replay all "2025/09/16 to 2025/09/18"
    #   python -m app.replay rain 2025/09/16 2025/09/18
    #   python -m app.replay wind_dir 2025-09-18
    if len(sys.argv) < 3:
        print("Usage: python -m app.replay <metric|all> <since> [until]")
        sys.exit(2)

    metric = sys.argv[1]
    if len(sys.argv) == 3:
        n = replay([metric], sys.argv[2], None)
    else:
        n = replay([metric], sys.argv[2], sys.argv[3])
    print(f"Inserted {n} alert(s).")
