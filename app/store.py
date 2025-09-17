import sqlite3
from contextlib import contextmanager
from typing import Iterable, Tuple
from .config import settings
from .log import logger

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS readings (
  ts TEXT NOT NULL,
  station_id TEXT NOT NULL,
  metric TEXT NOT NULL,
  value REAL,
  PRIMARY KEY (ts, station_id, metric)
);
CREATE INDEX IF NOT EXISTS idx_readings_metric_ts ON readings(metric, ts);
CREATE INDEX IF NOT EXISTS idx_readings_station ON readings(station_id);

CREATE TABLE IF NOT EXISTS scores (
  ts TEXT NOT NULL,
  station_id TEXT NOT NULL,
  metric TEXT NOT NULL,
  score REAL,
  method TEXT,
  extras_json TEXT,
  PRIMARY KEY (ts, station_id, metric, method)
);
CREATE INDEX IF NOT EXISTS idx_scores_metric_ts ON scores(metric, ts);

CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  station_id TEXT NOT NULL,
  metric TEXT NOT NULL,
  type TEXT NOT NULL,
  severity REAL,
  reason TEXT,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS stations (
  station_id TEXT PRIMARY KEY,
  name TEXT,
  lat REAL,
  lon REAL
);
"""

@contextmanager
def conn():
    c = sqlite3.connect(settings.database_path, check_same_thread=False)
    try:
        yield c
    finally:
        c.commit()
        c.close()


def init_db():
    with conn() as c:
        c.executescript(SCHEMA)
    logger.info("SQLite ready at {}", settings.database_path)


def write_readings(rows: Iterable[Tuple]):
    with conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO readings(ts, station_id, metric, value) VALUES (?,?,?,?)",
            rows,
        )


def write_scores(rows: Iterable[Tuple]):
    with conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO scores(ts, station_id, metric, score, method, extras_json) VALUES (?,?,?,?,?,?)",
            rows,
        )


def insert_alert(ts, station_id, metric, type_, severity, reason, payload_json):
    with conn() as c:
        c.execute(
            "INSERT INTO alerts(ts, station_id, metric, type, severity, reason, payload_json) VALUES (?,?,?,?,?,?,?)",
            (ts, station_id, metric, type_, severity, reason, payload_json),
        )


def get_latest_readings(metric: str, station_id: str = None, n: int = 300):
    with conn() as c:
        if station_id:
            cur = c.execute(
                "SELECT ts, station_id, metric, value FROM readings WHERE metric=? AND station_id=? ORDER BY ts DESC LIMIT ?",
                (metric, station_id, n),
            )
        else:
            cur = c.execute(
                "SELECT ts, station_id, metric, value FROM readings WHERE metric=? ORDER BY ts DESC LIMIT ?",
                (metric, n),
            )
        return cur.fetchall()


def get_alerts(metric: str = None, since: str = None):
    q = "SELECT id, ts, station_id, metric, type, severity, reason, payload_json FROM alerts WHERE 1=1"
    args = []
    if metric:
        q += " AND metric=?"; args.append(metric)
    if since:
        q += " AND ts>=?"; args.append(since)
    q += " ORDER BY ts DESC LIMIT 1000"
    with conn() as c:
        return c.execute(q, tuple(args)).fetchall()


def upsert_stations(rows: Iterable[Tuple]):
    with conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO stations(station_id, name, lat, lon) VALUES (?,?,?,?)",
            rows,
        )


def all_stations():
    with conn() as c:
        return c.execute("SELECT station_id, name, lat, lon FROM stations").fetchall()
