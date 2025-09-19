from fastapi import FastAPI, Query, HTTPException
from .store import conn, init_db, get_alerts, get_latest_readings, all_stations, get_active_stations, delete_alerts as _delete_alerts
from .stations import station_index
from .log import logger
from typing import Optional, List, Dict, Any

app = FastAPI(title="Weathering with Phew API")

# Ensure DB schema + hydrate station index at startup
@app.on_event("startup")
def _startup():
    init_db() 
    try:
        station_index._hydrate() # builds KDTree once station coordinates present
        logger.info("Station index hydrated")
    except Exception as e:
        logger.error(f"Station index hydrate failed: {e}")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/stations")
async def stations():
    rows = all_stations()
    return [{"station_id": r[0], "name": r[1], "lat": r[2], "lon": r[3]} for r in rows]

@app.get("/stations/active")
async def stations_active(
    minutes: int = Query(180, ge=1, le=1440),
    metric: str | None = Query(None, description="temperature|rainfall|humidity|wind_direction|wind_speed"),
):
    rows = get_active_stations(minutes=minutes, metric=metric)
    return [
        {"station_id": r[0], "name": r[1], "lat": r[2], "lon": r[3], "last_ts": r[4]}
        for r in rows
    ]

@app.get("/alerts")
async def alerts(
    metric: Optional[str] = Query(default=None),
    since: Optional[str] = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=5000),
) -> List[Dict[str, Any]]:
    try:
        rows = get_alerts(metric=metric, since=since, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not rows:
        return []

    out: List[Dict[str, Any]] = []

    # Returned dictionary
    if isinstance(rows[0], dict):
        for d in rows:
            d = dict(d)  # copy
            # Normalize payload
            if "payload" not in d and "payload_json" in d:
                pj = d.get("payload_json")
                try:
                    d["payload"] = json.loads(pj) if isinstance(pj, str) else pj
                except Exception:
                    d["payload"] = pj
            d.pop("payload_json", None)
            # Fallback station_name
            d.setdefault("station_name", d.get("station_id"))
            # Return only canonical fields (in order)
            out.append({
                "id": d.get("id"),
                "ts": d.get("ts"),
                "station_id": d.get("station_id"),
                "station_name": d.get("station_name"),
                "metric": d.get("metric"),
                "type": d.get("type"),
                "severity": d.get("severity"),
                "reason": d.get("reason"),
                "payload": d.get("payload"),
            })
        return out

    # Returned tuple
    # id, ts, station_id, station_name, metric, type, severity, reason, payload_json
    for r in rows:
        if len(r) == 9:
            rid, ts, station_id, station_name, metric_, type_, severity, reason, payload_json = r
        elif len(r) == 8:
            # No station_name column; use station_id
            rid, ts, station_id, metric_, type_, severity, reason, payload_json = r
            station_name = station_id
        else:
            # Unexpected shape — skip or raise
            raise HTTPException(status_code=500, detail=f"Unexpected alerts row len={len(r)}")

        try:
            payload = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
        except Exception:
            payload = payload_json

        out.append({
            "id": rid,
            "ts": ts,
            "station_id": station_id,
            "station_name": station_name,
            "metric": metric_,
            "type": type_,
            "severity": severity,
            "reason": reason,
            "payload": payload,
        })
    return out

@app.delete("/alerts")
async def delete_alerts_endpoint(
    metric: str | None = Query(None),
    since: str | None = Query(None),
    station_id: str | None = Query(None),
    type_: str | None = Query(None, alias="type"),
):
    deleted = _delete_alerts(metric=metric, since=since, station_id=station_id, type_=type_)
    return {"deleted": deleted}

@app.get("/latest")
async def latest(metric: str, station_id: str | None = None, n: int = 300):
    rows = get_latest_readings(metric, station_id, n)
    return [{"ts": r[0], "station_id": r[1], "metric": r[2], "value": r[3]} for r in rows]

@app.get("/")
async def index():
    return {
        "service": "Weathering with Phew API",
        "endpoints": ["/healthz", "/stations", "/stations/active", "/latest?metric=…", "/alerts", "/metrics", "/docs"]
    }

@app.get("/metrics")
async def metrics():
    with conn() as c:
        r = c.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        s = c.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        a = c.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    return {"readings": r, "scores": s, "alerts": a}
