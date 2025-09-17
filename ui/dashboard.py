import os
import json
import requests
import pandas as pd
import streamlit as st

API = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Weathering with Phew", layout="wide")
st.title("🌦️ Weathering with Phew")

# ------------ helpers ------------
@st.cache_data(ttl=10)
def api_get(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def load_stations():
    try:
        sts = api_get("/stations/active", minutes=180)
        if not sts:
            sts = api_get("/stations")
    except Exception:
        sts = []
    df = pd.DataFrame(sts)
    # id->name mapping for UI fallbacks
    id2name = {}
    if not df.empty and "station_id" in df and "name" in df:
        id2name = dict(zip(df["station_id"], df["name"]))
    return df, id2name

# ------------ Stations / selector ------------
col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("Stations")
    df_s, id2name = load_stations()

    if not df_s.empty and {"lat", "lon"}.issubset(df_s.columns):
        st.map(df_s.rename(columns={"lat": "latitude", "lon": "longitude"}))

    metric = st.selectbox(
        "Metric",
        ["temperature", "rainfall", "humidity", "wind_direction", "wind_speed"],
        index=0,
    )

    # stable list for selectbox; format_func shows human name
    station_ids = df_s["station_id"].tolist() if not df_s.empty else [""]
    station_id = st.selectbox(
        "Station",
        station_ids,
        format_func=lambda sid: id2name.get(sid, sid),
    )

    n = st.slider("Points", 60, 720, 300, step=30)

with col2:
    st.subheader("Latest readings")
    if station_id:
        try:
            data = api_get("/latest", metric=metric, station_id=station_id, n=n)
        except Exception:
            data = []
        df = pd.DataFrame(data)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            st.line_chart(df.set_index("ts")["value"], height=300)
        else:
            st.caption("No data yet — run daemon/collector.")

st.divider()

# ------------ Alerts ------------
st.subheader("Alerts")
metric_filter = st.selectbox(
    "Filter metric",
    ["", "temperature", "rainfall", "humidity", "wind_direction", "wind_speed"],
    index=0,
)
since = st.text_input("Since (ISO, optional)", "")

try:
    alerts = api_get("/alerts", metric=(metric_filter or None), since=(since or None))
except Exception:
    alerts = []

df_a = pd.DataFrame(alerts)
if not df_a.empty:
    # prefer API-provided station_name; fall back to mapping if missing
    if "station_name" not in df_a.columns:
        df_a["station_name"] = df_a.get("station_id", pd.Series([])).map(id2name)

    # parse payload JSON if it came as a string
    if "payload" in df_a.columns:
        def _parse_payload(p):
            if isinstance(p, dict):
                return p
            try:
                return json.loads(p) if isinstance(p, str) else p
            except Exception:
                return p
        df_a["payload"] = df_a["payload"].apply(_parse_payload)

    # tidy columns for display
    show_cols = [c for c in ["id", "ts", "station_name", "metric", "type", "severity", "reason", "payload"] if c in df_a.columns]
    df_a = df_a[show_cols].rename(columns={"station_name": "station"})
    st.dataframe(df_a.sort_values("ts", ascending=False), use_container_width=True, height=360)
else:
    st.caption("No alerts yet.")
