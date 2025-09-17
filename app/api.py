# app/api.py
from __future__ import annotations

import json
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from .store import conn, get_alerts, get_latest_readings, all_stations
from .log import logger

app = FastAPI(title="Weathering with Phew API")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

# ---- Stations ----

@app.get("/stations")
async def stations() -> List[Dict[str, Any]]:
    rows = all_stations()
    # rows: (id, name, lat, lon)
    return [{"station_id": r[0], "name": r[1], "lat": r[2], "lon": r[3]} for r in rows]

@app.get("/stations/active")
async def stations_active(
    metric: str = Query(..., description="Metric name (e.g. rainfall, temperature)"),
    minutes: int = Query(180, ge=1, le=1440, description="Lookback window in minutes"),
) -> List[Dict[str, Any]]:
    """
    Return only stations that have readings for the given metric in the last N minutes.
    """
    with conn() as c:
        ids = [r[0] for r in c.execute(
            "SELECT DISTINCT station_id FROM readings "
            "WHERE metric=? AND ts >= datetime('now', ?) ",
            (metric, f"-{minutes} minutes"),
        ).fetchall()]
        if not ids:
            return []
        q = "SELECT id, name, lat, lon FROM stations WHERE id IN (%s)" % ",".join(["?"] * len(ids))
        st = c.execute(q, ids).fetchall()
    return [{"station_id": r[0], "name": r[1], "lat": r[2], "lon": r[3]} for r in st]

# ---- Alerts ----

@app.get("/alerts")
async def alerts(
    metric: Optional[str] = None,
    since: Optional[str] = Query(None, description="ISO datetime; returns alerts at/after this time"),
    limit: int = Query(500, ge=1, le=5000),
) -> List[Dict[str, Any]]:
    """
    Return alerts (newest first). Includes station_name and parsed payload.
    """
    # Prefer doing the join here so every caller gets station_name without extra calls
    where = []
    params: List[Any] = []
    if metric:
        where.append("a.metric = ?")
        params.append(metric)
    if since:
        # basic validation
        try:
            # Let SQLite compare ISO strings fine; we just sanity-check format here
            _ = since.replace("Z", "")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid 'since': {e}")
        where.append("a.ts >= ?")
        params.append(since)

    sql = f"""
    SELECT a.id, a.ts, a.station_id, COALESCE(s.name, a.station_id) AS station_name,
           a.metric, a.type, a.severity, a.reason, a.payload_json
    FROM alerts a
    LEFT JOIN stations s ON s.id = a.station_id
    {"WHERE " + " AND ".join(where) if where else ""}
    ORDER BY a.ts DESC
    LIMIT ?
    """
    params.append(limit)

    with conn() as c:
        rows = c.execute(sql, params).fetchall()

    out = []
    for r in rows:
        payload = None
        if r[8]:
            try:
                payload = json.loads(r[8])
            except Exception:
                payload = r[8]  # keep raw if not valid JSON
        out.append({
            "id": r[0],
            "ts": r[1],
            "station_id": r[2],
            "station_name": r[3],
            "metric": r[4],
            "type": r[5],
            "severity": r[6],
            "reason": r[7],
            "payload": payload,
        })
    return out

# ---- Latest readings ----

@app.get("/latest")
async def latest(
    metric: str = Query(..., description="Metric name"),
    station_id: Optional[str] = Query(None, description="Station ID; if omitted returns mixed-station recent rows"),
    n: int = Query(300, ge=1, le=2000),
) -> List[Dict[str, Any]]:
    rows = get_latest_readings(metric, station_id, n)
    # rows: (ts, station_id, metric, value)
    return [{"ts": r[0], "station_id": r[1], "metric": r[2], "value": r[3]} for r in rows]

# ---- Metrics / Index ----

@app.get("/metrics")
async def metrics():
    with conn() as c:
        r = c.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        s = c.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        a = c.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    return {"readings": r, "scores": s, "alerts": a}

@app.get("/")
async def index():
    return {
        "service": "Weathering with Phew API",
        "endpoints": [
            "/healthz",
            "/stations",
            "/stations/active?metric=…&minutes=…",
            "/latest?metric=…&station_id=…&n=…",
            "/alerts?metric=…&since=…&limit=…",
            "/metrics",
            "/docs",
        ],
    }
