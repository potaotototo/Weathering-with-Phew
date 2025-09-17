import pandas as pd
from app.rules import apply_rules


def test_sustained_triggers(tmp_path, monkeypatch):
    # create a fake window for one station with sustained high z
    ts = pd.date_range("2024-01-01", periods=10, freq="min")
    df = pd.DataFrame({
        "ts": ts,
        "station_id": ["S_1"]*10,
        "metric": ["temperature"]*10,
        "value": [30]*10,
        "z": [4]*10,
        "score": [0.9]*10,
        "neighbor_flag": [True]*10,
        "neighbor_gap": [1.2]*10
    })
    # monkeypatch DB insert to capture alerts instead of hitting SQLite
    captured = []
    from app import rules
    def fake_insert(ts, station_id, metric, type_, severity, reason, payload_json):
        captured.append((ts, station_id, metric, type_, severity))
    monkeypatch.setattr(rules, "insert_alert", fake_insert)

    df["ts"] = pd.to_datetime(df["ts"]) 
    out = apply_rules(df, "temperature", df["ts"].iloc[-1])
    assert captured, "Expected at least one alert"
    assert captured[0][3] == "WEATHER"
