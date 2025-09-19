import os
import json
import math
import requests
import pandas as pd
import streamlit as st

from requests.exceptions import ReadTimeout

API = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Weathering with Phew", layout="wide")
st.title("🌦️ Weathering with Phew")

# Helpers
@st.cache_data(ttl=10)
def api_get(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def api_delete(path: str, **params):
    r = requests.delete(f"{API}{path}", params=params, timeout=30) 
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=10)
def load_stations_server(minutes: int, metric: str | None):
    """Always try server-side active endpoint first."""
    params = {"minutes": minutes}
    if metric:
        params["metric"] = metric
    data = api_get("/stations/active", **params) # raises on HTTP != 200
    df = pd.DataFrame(data)
    return df

@st.cache_data(ttl=60)
def load_stations_all():
    return pd.DataFrame(api_get("/stations"))

def id_to_name_map(df):
    return dict(zip(df.get("station_id", []), df.get("name", [])))

# UI main body
st.divider()
col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("Select Available Stations")

    active_only = st.checkbox("Show active stations only", value=True)
    active_minutes = st.slider(
        "Active window (minutes)", 30, 360, 180, step=15, disabled=not active_only
    )

    # Scope for "active" (any metric vs selected metric)
    # This radio lives in the left column so the list updates when change is detected
    active_scope = st.radio(
        "Active scope",
        ["Any metric", "Selected metric"],
        index=0,
        horizontal=True,
        disabled=not active_only,
        help="Use 'Selected metric' to only show stations active for the chosen metric."
    )

with col2:
    st.subheader("View Readings")
    metric = st.selectbox(
        "Metric",
        ["temperature", "rainfall", "humidity", "wind_direction", "wind_speed"],
        index=0,
    )
    n = st.slider("Points", 60, 720, 300, step=30)

# Load stations (prefer server)
try:
    if active_only:
        metric_for_active = None if active_scope == "Any metric" else metric
        df_s = load_stations_server(active_minutes, metric_for_active)
        source_label = "active via server"
    else:
        df_s = load_stations_all()
        source_label = "all stations"
except ReadTimeout:
    # Fast fallback if active endpoint is temporarily slow (e.g., backfill running)
    df_s = load_stations_all()
    source_label = "all stations (active endpoint timed out)"
    st.warning("Active-stations endpoint timed out — showing all stations for now.")
except Exception:
    df_s = load_stations_all()
    source_label = "all stations (active endpoint unavailable)"

id2name = id_to_name_map(df_s)

with col1:
    # Map
    if not df_s.empty and {"lat", "lon"}.issubset(df_s.columns):
        st.map(df_s.rename(columns={"lat": "latitude", "lon": "longitude"}))
    st.caption(f"Stations: showing {len(df_s)} ({source_label})")

with col2:
    # Station picker
    station_ids = df_s["station_id"].tolist() if not df_s.empty else [""]
    station_id = st.selectbox(
        "Station",
        station_ids,
        format_func=lambda sid: id2name.get(sid, sid)
    )

    st.subheader("Latest readings")
    if station_id:
        try:
            data = api_get("/latest", metric=metric, station_id=station_id, n=n)
        except Exception:
            data = []
        df = pd.DataFrame(data)
        if not df.empty:
            df["ts"] = (pd.to_datetime(df["ts"], utc=True, errors="coerce")
              .dt.tz_convert("Asia/Singapore"))
            st.line_chart(df.set_index("ts")["value"], height=300)
        else:
            st.caption("No data yet — run daemon/collector.")

st.divider()

# Alerts with location filters

st.subheader("Alerts")

metric_filter = st.selectbox(
    "Filter metric",
    ["", "temperature", "rainfall", "humidity", "wind_direction", "wind_speed"],
    index=0,
)
since = st.text_input("Since (ISO, optional)", "")

with st.expander("Location filter", expanded=False):
    mode = st.radio("Mode", ["None", "Near a station", "Bounding box"], horizontal=True)

    bbox = None
    center_sid, radius_km = None, None

    if mode == "Bounding box":
        st.caption("Show alerts where station coordinates fall within the rectangle.")
        c1, c2 = st.columns(2)
        with c1:
            min_lat = st.number_input("Min latitude", value=1.200000, step=0.001, format="%.6f")
            min_lon = st.number_input("Min longitude", value=103.600000, step=0.001, format="%.6f")
        with c2:
            max_lat = st.number_input("Max latitude", value=1.485000, step=0.001, format="%.6f")
            max_lon = st.number_input("Max longitude", value=104.100000, step=0.001, format="%.6f")
        bbox = (min_lat, max_lat, min_lon, max_lon)

    if mode == "Near a station":
        st.caption("Show alerts within a radius of a station.")
        center_sid = st.selectbox(
            "Center station",
            df_s["station_id"].tolist() if not df_s.empty else [""],
            format_func=lambda sid: id2name.get(sid, sid),
        )
        radius_km = st.slider("Radius (km)", 1, 25, 5, step=1)

# Pull alerts from API, filter by metric/since server-side
try:
    alerts = api_get("/alerts", metric=(metric_filter or None), since=(since or None))
except Exception:
    alerts = []

df_alerts = pd.DataFrame(alerts)

if df_alerts.empty:
    st.caption("No alerts yet.")
else:
    # Display SG timezone
    df_alerts["ts"] = (
        pd.to_datetime(df_alerts["ts"], utc=True, errors="coerce")
          .dt.tz_convert("Asia/Singapore")
    )

    # Ensure station_name; join lat/lon for location filtering
    if "station_name" not in df_alerts.columns:
        df_alerts["station_name"] = df_alerts.get("station_id").map(id2name)
    if not df_s.empty and {"station_id", "lat", "lon"}.issubset(df_s.columns):
        df_alerts = df_alerts.merge(df_s[["station_id", "lat", "lon"]], on="station_id", how="left")

    # Apply location filters (client-side)
    df_filtered = df_alerts.copy()

    if mode == "Bounding box" and bbox and {"lat", "lon"}.issubset(df_filtered.columns):
        min_lat, max_lat, min_lon, max_lon = bbox
        df_filtered = df_filtered[
            df_filtered["lat"].between(min_lat, max_lat) &
            df_filtered["lon"].between(min_lon, max_lon)
        ]

    if mode == "Near a station" and center_sid and radius_km and not df_s.empty:
        row = df_s[df_s["station_id"] == center_sid]
        if not row.empty and {"lat", "lon"}.issubset(row.columns):
            c_lat, c_lon = float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])

            def _haversine_km(lat1, lon1, lat2, lon2):
                R = 6371.0
                import math
                p1, p2 = math.radians(lat1), math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlmb = math.radians(lon2 - lon1)
                a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
                return 2*R*math.asin(math.sqrt(a))

            def _dist_ok(r):
                try:
                    return _haversine_km(c_lat, c_lon, float(r["lat"]), float(r["lon"])) <= radius_km
                except Exception:
                    return False

            df_filtered = df_filtered[df_filtered.apply(_dist_ok, axis=1)]

    # Show selected payload details
    def _parse_payload(p):
        if isinstance(p, dict):
            return p
        try:
            return json.loads(p) if isinstance(p, str) else p
        except Exception:
            return p

    def _payload_summary(p: object, max_items: int = 4, max_len: int = 120) -> str:
        if isinstance(p, dict):
            parts = []
            for k in sorted(p.keys()):
                v = p[k]
                if isinstance(v, float):
                    v = round(v, 3)
                elif isinstance(v, (list, dict)):
                    v = "[…]"
                parts.append(f"{k}={v}")
                if len(parts) >= max_items:
                    break
            s = ", ".join(parts)
        else:
            s = str(p)
        return (s[:max_len] + "…") if len(s) > max_len else s

    # Parse payload from either 'payload' or 'payload_json'
    if "payload" in df_filtered.columns:
        df_filtered["payload"] = df_filtered["payload"].apply(_parse_payload)
    elif "payload_json" in df_filtered.columns:
        df_filtered["payload"] = df_filtered["payload_json"].apply(_parse_payload)

    # Add compact summary for table
    df_filtered["payload_summary"] = df_filtered.get("payload", pd.Series([None]*len(df_filtered))).apply(_payload_summary)

    # Display
    show_cols = [c for c in [
        "id", "ts", "station_name", "metric", "type", "severity", "reason", "lat", "lon", "payload_summary"
    ] if c in df_filtered.columns]
    df_disp = df_filtered[show_cols].rename(columns={"station_name": "station", "payload_summary": "payload"})

    st.caption("Times shown in SGT; stored in UTC.")
    st.caption(f"Showing {len(df_disp)} of {len(df_alerts)} alert(s)")

    st.dataframe(df_disp.sort_values("ts", ascending=False), use_container_width=True, height=380)

    # Details panel for full JSON payload
    with st.expander("🔎 Payload details", expanded=False):
        if "id" in df_filtered.columns and not df_filtered.empty:
            # Default to most recent alert
            recent = df_filtered.sort_values("ts", ascending=False).reset_index(drop=True)
            sel_id = st.selectbox("Select alert", options=recent["id"].tolist(), index=0)
            row = df_filtered[df_filtered["id"] == sel_id].iloc[0]
            st.write(f"**{row.get('station_name', row.get('station_id'))}** • "
                     f"{row.get('metric')} • {row.get('type')} • {row.get('ts')}")
            st.json(row.get("payload", {}), expanded=True)
            st.code(json.dumps(row.get("payload", {}), ensure_ascii=False, indent=2), language="json")
        else:
            st.caption("No alerts to show.")

st.divider()

# Clear alerts (with filters)

st.subheader("Clear Alerts")

with st.expander("🧹 Admin: Clear alerts", expanded=False):
    # pull stations for nicer dropdown (if not already loaded earlier)
    try:
        stations_admin = api_get("/stations/active", minutes=180)
        if not stations_admin:
            stations_admin = api_get("/stations")
    except Exception:
        stations_admin = []
    df_admin_s = pd.DataFrame(stations_admin)
    id2name_admin = dict(zip(df_admin_s.get("station_id", []), df_admin_s.get("name", [])))

    c1, c2, c3, c4 = st.columns([1,1,1,1])

    with c1:
        metric_del = st.selectbox(
            "Metric (optional)",
            ["", "temperature", "rainfall", "humidity", "wind_direction", "wind_speed"],
            index=0,
            key="del_metric",
        )
    with c2:
        station_del = st.selectbox(
            "Station (optional)",
            [""] + (df_admin_s["station_id"].tolist() if not df_admin_s.empty else []),
            format_func=lambda sid: id2name_admin.get(sid, sid),
            key="del_station",
        )
    with c3:
        since_del = st.text_input("Since (ISO, optional)", "", key="del_since")
    with c4:
        type_del = st.text_input("Type (optional)", "", key="del_type")

    if st.button("Delete matching alerts", type="primary"):
        try:
            resp = api_delete(
                "/alerts",
                metric=(metric_del or None),
                since=(since_del or None),
                station_id=(station_del or None),
                type=(type_del or None),
            )
            st.success(f"Deleted {resp.get('deleted', 0)} alert(s).")
            st.cache_data.clear()  # refreshes alerts table next rerun
        except Exception as e:
            st.error(f"Delete failed: {e}")
