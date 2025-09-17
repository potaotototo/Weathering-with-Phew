from fastapi import FastAPI, Query
from .store import conn, init_db, get_alerts, get_latest_readings, all_stations, get_active_stations
from .stations import station_index
from .log import logger

app = FastAPI(title="Weathering with Phew API")

# --- ensure DB schema + hydrate station index at startup ---
@app.on_event("startup")
def _startup():
    init_db()                 # creates tables / WAL if not present (idempotent)
    try:
        station_index._hydrate()  # builds KDTree once we have station coords
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
async def alerts(metric: str | None = None, since: str | None = None):
    rows = get_alerts(metric=metric, since=since)
    return [
        {
            "id": r[0], "ts": r[1], "station_id": r[2],
            "metric": r[3], "type": r[4], "severity": r[5],
            "reason": r[6], "payload": r[7]
        }
        for r in rows
    ]

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
