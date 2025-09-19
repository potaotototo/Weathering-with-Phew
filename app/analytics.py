from __future__ import annotations
from typing import Iterable, Mapping, Optional

from sqlalchemy import (
    create_engine, MetaData, Table, select, func, case, literal, text
)
from sqlalchemy.engine import Engine

def make_engine(db_path: str = "weatherguard.db") -> Engine:
    # Uses the modern SQLite driver name
    return create_engine(f"sqlite+pysqlite:///{db_path}", future=True)

# Reflect existing tables (no ORM models needed)
def reflect_tables(engine: Engine):
    md = MetaData()
    readings = Table("readings", md, autoload_with=engine)
    stations = Table("stations", md, autoload_with=engine)
    return readings, stations

# Last N non-zero rainfall ticks (SGT) 
def recent_rain_ticks(engine: Engine, limit: int = 200, days: Optional[int] = None) -> list[Mapping]:
    readings, stations = reflect_tables(engine)

    ts_sgt = func.datetime(readings.c.ts, text("'+8 hours'")).label("ts_sgt")
    station = func.coalesce(stations.c.name, readings.c.station_id).label("station")

    q = (
        select(ts_sgt, station, readings.c.value.label("mm_5min"))
        .select_from(readings.outerjoin(stations, readings.c.station_id == stations.c.station_id))
        .where(readings.c.metric == "rainfall", readings.c.value > 0)
        .order_by(readings.c.ts.desc())
        .limit(limit)
    )
    if days:
        q = q.where(readings.c.ts >= func.datetime("now", f"-{days} days"))

    with engine.begin() as conn:
        return list(conn.execute(q).mappings())

# Daily station totals (SGT) for recent window 
def daily_rain_totals(engine: Engine, days: int = 14) -> list[Mapping]:
    readings, stations = reflect_tables(engine)

    day_sgt = func.date(func.datetime(readings.c.ts, text("'+8 hours'"))).label("day_sgt")
    station = func.coalesce(stations.c.name, readings.c.station_id).label("station")
    mm_total = func.round(func.sum(readings.c.value), 2).label("mm_total")

    q = (
        select(day_sgt, station, mm_total)
        .select_from(readings.join(stations, readings.c.station_id == stations.c.station_id))
        .where(readings.c.metric == "rainfall",
               readings.c.ts >= func.datetime("now", f"-{days} days"))
        .group_by(day_sgt, station)
        .having(func.sum(readings.c.value) > 0)
        .order_by(day_sgt.desc(), mm_total.desc())
    )

    with engine.begin() as conn:
        return list(conn.execute(q).mappings())

# Continuous “rain episodes” using window functions
def rain_episodes(engine: Engine, days: int = 7) -> list[Mapping]:
    readings, stations = reflect_tables(engine)

    # base selection
    base = (
        select(
            readings.c.ts,
            readings.c.station_id,
            readings.c.value,
            func.coalesce(stations.c.name, readings.c.station_id).label("station_name"),
            case((readings.c.value > 0, 1), else_=0).label("wet"),
        )
        .select_from(readings.join(stations, readings.c.station_id == stations.c.station_id))
        .where(readings.c.metric == "rainfall",
               readings.c.ts >= func.datetime("now", f"-{days} days"))
        .subquery("rain")
    )

    # mark starts of wet spells
    wet_prev = func.lag(base.c.wet).over(
        partition_by=base.c.station_id, order_by=base.c.ts
    ).label("wet_prev")

    marks = select(base, wet_prev).subquery("marks")

    is_start = case(
        ((marks.c.wet == 1) & (func.coalesce(marks.c.wet_prev, 0) == 0), 1),
        else_=0,
    ).label("is_start")

    starts = select(marks, is_start).subquery("starts")

    # running group id per station
    grp_id = func.sum(starts.c.is_start).over(
        partition_by=starts.c.station_id, order_by=starts.c.ts
    ).label("grp_id")

    grp = select(starts, grp_id).where(starts.c.wet == 1).subquery("grp")

    start_sgt = func.min(func.datetime(grp.c.ts, text("'+8 hours'"))).label("start_sgt")
    end_sgt   = func.max(func.datetime(grp.c.ts, text("'+8 hours'"))).label("end_sgt")
    mm_total  = func.round(func.sum(grp.c.value), 2).label("mm_total")
    minutes_wet = (func.count() * literal(5)).label("minutes_wet")

    q = (
        select(grp.c.station_name, start_sgt, end_sgt, mm_total, minutes_wet)
        .group_by(grp.c.station_id, grp.c.grp_id)
        .order_by(start_sgt.desc())
    )

    with engine.begin() as conn:
        return list(conn.execute(q).mappings())

if __name__ == "__main__":
    engine = make_engine("weatherguard.db")

    print(recent_rain_ticks(engine, limit=50, days=7)[:3])
    print(daily_rain_totals(engine, days=14)[:5])
    print(rain_episodes(engine, days=7)[:5])
