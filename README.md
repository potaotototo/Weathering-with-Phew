"""
# Weathering with Phew — MVP

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "from app.store import init_db; init_db()"

# (Optional) export your data.gov.sg API key for higher rate limits
export DATA_GOV_SG_API_KEY=YOUR_SECRET_TOKEN

# First, run a one-shot collector call to populate stations & readings (latest)
python - <<'PY'
from app.collector import poll_once
poll_once(metrics=("temperature",))
PY

# Backfill an entire day if you want more data (uses pagination)
python - <<'PY'
from app.collector import backfill_day
print("rows temp:", backfill_day("temperature", "2024-07-16"))
print("rows rain:", backfill_day("rainfall", "2024-07-16"))
print("rows humid:", backfill_day("humidity", "2024-07-16"))
print("rows winddir:", backfill_day("wind_direction", "2024-07-16"))
print("rows windspd:", backfill_day("wind_speed", "2024-07-16"))
PY

# Serve API
uvicorn app.api:app --reload --port 8000

# Run the engine periodically (or cron); one-shot example
python - <<'PY'
from app.engine import Engine
from app.stations import station_index
station_index._hydrate()  # load stations from DB
Engine().process_tick(None)
PY

# Streamlit UI
streamlit run ui/dashboard.py
```

## Notes
- New NEA endpoint base: `https://api-open.data.gov.sg/v2/real-time/api/air-temperature` with `date` and `paginationToken` support.
- Collector adapted to new schema: `data.stations[*]` (uses `labelLocation` lat/lon) and `data.readings[*]` blocks with `timestamp` + `data[] {stationId, value}`.
- Add remaining endpoints (rainfall/humidity/wind) to `settings.nea_endpoints` when you share them; everything else is already pluggable.
"""bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "from app.store import init_db; init_db()"
# First, run a one-shot collector call to populate stations & readings
python -c "from app.collector import poll_once; poll_once()"
# Serve API
uvicorn app.api:app --reload --port 8000
# In another terminal, run the engine periodically (or cron)
# (Optional minimal runner) — you can wire this into a daemon later
python - <<'PY'
from app.engine import Engine
from app.stations import station_index
from app.store import init_db
station_index._hydrate()  # load stations from DB
eng = Engine()
eng.process_tick(None)
PY
# Streamlit UI
streamlit run ui/dashboard.py
```

## Notes
- The collector hits NEA latest endpoints and writes to SQLite; run it every minute via cron or a simple loop.
- `Engine.process_tick` computes features, neighbor gaps, scores (IsolationForest fallback to |z| when cold), and writes scores + alerts.
- FastAPI exposes `/stations`, `/latest`, `/alerts` for the UI.
- Tight, swappable `model.score()` interface for later LSTM-AE.
"""
