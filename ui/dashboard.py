import streamlit as st
import pandas as pd
import requests
import os
API = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Weathering with Phew", layout="wide")

st.title("🌦️ Weathering with Phew")

col1, col2 = st.columns([2,3])
with col1:
    st.subheader("Stations")
    try:
        stations = requests.get(f"{API}/stations").json()
    except Exception:
        stations = []
    df_s = pd.DataFrame(stations)
    if not df_s.empty and {"lat","lon"}.issubset(df_s.columns):
        st.map(df_s.rename(columns={"lat":"latitude","lon":"longitude"}))
    metric = st.selectbox("Metric", ["temperature","rainfall","humidity","wind_direction","wind_speed"], index=0)
    station_id = st.selectbox("Station", df_s.station_id.tolist() if not df_s.empty else [""])
    n = st.slider("Points", 60, 720, 300, step=30)

with col2:
    st.subheader("Latest readings")
    if station_id:
        try:
            data = requests.get(f"{API}/latest", params={"metric": metric, "station_id": station_id, "n": n}).json()
        except Exception:
            data = []
        df = pd.DataFrame(data)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"]) 
            st.line_chart(df.set_index("ts")["value"], height=300)
        else:
            st.caption("No data yet — run collector.")

st.divider()

st.subheader("Alerts")
metric_filter = st.selectbox("Filter metric", ["", "temperature","rainfall","humidity","wind_direction","wind_speed"], index=0)
since = st.text_input("Since (ISO)")
try:
    alerts = requests.get(f"{API}/alerts", params={"metric": metric_filter or None, "since": since or None}).json()
except Exception:
    alerts = []
df_a = pd.DataFrame(alerts)
if not df_a.empty:
    st.dataframe(df_a)
else:
    st.caption("No alerts yet.")
    
@st.cache_data(ttl=10)
def api_get(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()
