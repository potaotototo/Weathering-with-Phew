import requests

from datetime import datetime, timezone
from .config import settings
from .store import write_readings, upsert_stations
from .log import logger
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

_sess = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry)
_sess.mount("https://", _adapter)
_sess.mount("http://", _adapter)

def _headers():
    h = {"User-Agent": "Weathering-with-Phew/0.2"}
    if settings.x_api_key:
        h["X-Api-Key"] = settings.x_api_key 
    return h

def _fetch_metric(metric: str, date: str | None = None, pagination_token: str | None = None):
    if metric not in settings.nea_endpoints:
        raise ValueError(f"Unknown metric '{metric}'. Configure in settings.nea_endpoints")
    url = f"{settings.nea_base_url}/{settings.nea_endpoints[metric]}"
    params = {}
    if date:
        params["date"] = date # YYYY-MM-DD or YYYY-MM-DDTHH:mm:ss (SGT)
    if pagination_token:
        params["paginationToken"] = pagination_token
    r = _sess.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _parse_rt_schema(j: dict, metric: str):
    """
    Output rows: (ts_norm, station_id, metric, value)
    Also returns station rows: (id, name, lat, lon)
    """
    data = j.get("data") or {}
    stations = data.get("stations", [])
    readings = data.get("readings", [])

    # station metadata
    station_rows = []
    for s in stations:
        loc = s.get("labelLocation") or s.get("location") or {}
        lat = loc.get("latitude"); lon = loc.get("longitude")
        station_rows.append((
            s.get("id"),
            s.get("name") or s.get("id"),
            float(lat) if lat is not None else None,
            float(lon) if lon is not None else None,
        ))

    # readings
    rows = []
    for blk in readings:
        ts_iso = blk.get("timestamp")
        if not ts_iso:
            continue
        ts_norm = normalize_ts(ts_iso)
        for d in blk.get("data", []):
            sid = d.get("stationId")
            val = d.get("value")
            if sid is None or val is None:
                continue
            rows.append((ts_norm, sid, metric, float(val)))

    # pagination
    next_token = data.get("paginationToken")
    return rows, station_rows, next_token

def poll_once(metrics=("temperature","rainfall","humidity","wind_direction","wind_speed"), date: str | None = None):
    """Fetch latest (date=None) or a page for a given date for the listed metrics."""
    all_rows = []
    st_rows = []
    for m in metrics:
        try:
            j = _fetch_metric(m, date=date)
            rows, stations, _ = _parse_rt_schema(j, m)
            all_rows.extend(rows)
            st_rows.extend(stations)
        except Exception as e:
            logger.error(f"Fetch {m} failed: {e}")
    if st_rows:
        # filter out Nones to avoid KDTree issues later
        st_rows = [r for r in st_rows if r[2] is not None and r[3] is not None]
        upsert_stations(st_rows)
    if all_rows:
        # drop rows missing station_id, ts, or value
        all_rows = [r for r in all_rows if r[0] and r[1] and (r[3] is not None)]
        write_readings(all_rows)
        logger.info("Ingested {} readings ({} metrics){}", len(all_rows), len(metrics), f" for date={date}" if date else "")
    return len(all_rows)

def backfill_day(metric: str, date: str):
    """Page through the entire day using paginationToken. date format: YYYY-MM-DD."""
    total = 0
    next_token = None
    while True:
        j = _fetch_metric(metric, date=date, pagination_token=next_token)
        rows, stations, next_token = _parse_rt_schema(j, metric)
        if stations:
            stations = [r for r in stations if r[2] is not None and r[3] is not None]
            upsert_stations(stations)
        if rows:
            rows = [r for r in rows if r[0] and r[1] and (r[3] is not None)]
            write_readings(rows)
            total += len(rows)
            logger.info("Backfill {} {}: +{} (cum {})", metric, date, len(rows), total)
        if not next_token:
            break
    return total

def backfill_range(metric: str, start_date: str, end_date: str) -> int:
    """
    Backfill inclusive date range [start_date, end_date], format 'YYYY-MM-DD' (SGT).
    Returns total rows ingested across the range.
    """
    d0 = date.fromisoformat(start_date)
    d1 = date.fromisoformat(end_date)
    if d1 < d0:
        d0, d1 = d1, d0
    total = 0
    cur = d0
    while cur <= d1:
        total += backfill_day(metric, cur.isoformat())
        cur += timedelta(days=1)
    return total

def backfill_week(metric: str, end_date: str | None = None, days: int = 7) -> int:
    """
    Backfill the last `days` days ending at `end_date` (inclusive).
    - end_date: 'YYYY-MM-DD' in SGT; if None, uses 'today' in SGT.
    - days: number of days to backfill (default 7).
    """
    if end_date:
        end_d = date.fromisoformat(end_date)
    else:
        if ZoneInfo:
            end_d = datetime.now(ZoneInfo("Asia/Singapore")).date()
        else:
            # fallback: local 'today'
            end_d = datetime.now().date()

    total = 0
    for i in range(days):
        d = end_d - timedelta(days=i)
        total += backfill_day(metric, d.isoformat())
    return total

def normalize_ts(ts: str) -> str:
    """
    Normalize NEA timestamps to SQLite-friendly UTC 'YYYY-MM-DD HH:MM:SS'.
    Accepts:
      - '2024-07-16T15:59:00.000Z'
      - '2024-07-16T15:59:00Z'
      - '2024-07-16T15:59:00+08:00'
      - '2024-07-16T15:59:00.123456+08:00'
    """
    if not ts:
        raise ValueError("empty timestamp")
    s = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # try dropping fractional seconds if parsing fails
        if "." in s:
            head, rest = s.split(".", 1)
            tz = ""
            for sep in ("+", "-"):
                if sep in rest:
                    tz = sep + rest.split(sep, 1)[1]
                    break
            dt = datetime.fromisoformat(head + (tz or ""))
        else:
            raise
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S")
