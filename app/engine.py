from queue import Queue
from threading import Thread
import time 
from datetime import datetime
import pandas as pd
import json

from .config import settings
from .store import get_latest_readings, write_scores, get_readings_since
from .stations import station_index
from .features import (
    compute_univariate_feats,
    neighbor_gap,
    compute_tod_baseline,
    attach_tod_residuals,
)
from .model import IsoForestModel, MultiIsoForest
from .rules import apply_simple_delta_rules

# Import check
try:
    from .rules import apply_rain_event_rules
except Exception:
    apply_rain_event_rules = None
try:
    from .rules import apply_wind_speed_rules
except Exception:
    apply_wind_speed_rules = None
try:
    from .rules import apply_temp_tod_rules
except Exception:
    apply_temp_tod_rules = None

from .log import logger

class Engine:
    def __init__(self):
        self.q = Queue(maxsize=100)
        self.model = MultiIsoForest()
        self._temp_tod_ref = None # cache
        self._temp_tod_last_build = None # cache

    def _ensure_temp_baseline(self):
        """Build/rebuild minute-of-day baseline for temperature using last N days.
        Rebuild at most once per hour."""
        if self._temp_tod_last_build and (datetime.utcnow() - self._temp_tod_last_build) < timedelta(hours=1):
            return self._temp_tod_ref is not None

        lookback = settings.tod_baseline_lookback_days
        since = (datetime.utcnow() - timedelta(days=lookback)).strftime("%Y-%m-%d %H:%M:%S")
        rows = get_readings_since("temperature", since)
        if not rows or len(rows) < 500:
            self._temp_tod_ref = None
            self._temp_tod_last_build = datetime.utcnow()
            return False

        df_hist = pd.DataFrame(rows, columns=["ts","station_id","metric","value"])
        df_hist["ts"] = pd.to_datetime(df_hist["ts"], utc=True)
        self._temp_tod_ref = compute_tod_baseline(df_hist, lookback_days=lookback)
        self._temp_tod_last_build = datetime.utcnow()
        return True


    def producer(self):
        # simple tick producer (each minute); in real usage, trigger after collector writes
        while True:
            self.q.put(datetime.utcnow())
            time.sleep(settings.poll_interval_seconds)

    def process_tick(self, ts: datetime):
        """
        Pull enough recent rows per metric to cover the time window for all stations,
        compute features, add neighbor_gap on the latest snapshot, score, and apply rules.
        """
        window = settings.rolling_window_minutes
        cadence_min = 5 # NEA realtime cadence
        # ~ window/cadence samples per station + a little buffer for edges
        samples_per_station = max(6, window // cadence_min + 4)

        # Estimate station count from KDTree (fallback to 1)
        try:
            station_count = max(1, len(getattr(station_index, "_stations", {})) or 1)
        except Exception:
            station_count = 1

        n_per_metric = station_count * samples_per_station

        # 1) Load recent readings for all metrics
        records = []
        for metric in ("temperature", "rainfall", "humidity", "wind_direction", "wind_speed"):
            rows = get_latest_readings(metric, station_id=None, n=n_per_metric)
            for ts_, sid, m, val in rows:
                records.append((pd.to_datetime(ts_), sid, m, float(val)))

        if not records:
            return

        df = pd.DataFrame(records, columns=["ts", "station_id", "metric", "value"]).dropna()

        # 2) Rolling features (per station, per metric)
        feats = compute_univariate_feats(df, window)
        if feats.empty:
            return

        # 3) KDTree neighbor map
        neighbors_map = {
            sid: station_index.neighbors(sid, settings.neighbor_k)
            for sid in feats.station_id.unique()
        }

        # 4) Latest row per (metric, station) + neighbor_gap there
        latest = feats.sort_values("ts").groupby(["metric", "station_id"]).tail(1)

        # 5) Fit/score per metric, apply rules
        scored_metrics = 0
        for metric, g in latest.groupby("metric"):
            # compute neighbor_gap using ONLY this metric’s snapshot
            g = neighbor_gap(g, neighbors_map)
            feat_m = feats[feats.metric == metric]
            if feat_m.empty or g.empty:
                continue

            # Train on historical feats (neighbor_gap not required for training)
            self.model.fit_if_needed(metric, feat_m)

            g = g.copy()
            # Score NEEDS neighbor_gap -> present in 'g'
            g["score"] = self.model.score(metric, g)

            # Neighbor coherence from this timestamp snapshot (kept for observability)
            idx = g.set_index("station_id")
            nb_flag, nb_count = [], []
            for sid in g.station_id:
                nb_ids = [n for n, _ in neighbors_map.get(sid, [])]
                sub = idx.loc[idx.index.intersection(nb_ids)]
                if sub.empty:
                    nb_flag.append(False); nb_count.append(0); continue
                anoms = ((sub["z"].abs() > settings.neighbor_z_threshold) | (sub["score"] > 0.7))
                nb_count.append(int(anoms.sum()))
                nb_flag.append(bool(anoms.any()))
            g["neighbor_flag"] = nb_flag
            g["neighbor_count_anom"] = nb_count

            # Persist scores
            score_rows = [
                (str(r.ts), r.station_id, metric, float(r.score), "isoforest",
                 json.dumps({
                     "z": float(r.z) if pd.notna(r.z) else None,
                     "neighbor_gap": float(r.neighbor_gap) if pd.notna(r.neighbor_gap) else None
                 }))
                for r in g.itertuples()
            ]
            write_scores(score_rows)

            # Rules
            # Use only raw values/timestamps for rule checks
            # inside for metric, g in latest.groupby("metric"):
            window_df = feat_m[["ts", "station_id", "metric", "value"]].copy()
            t_cur = pd.to_datetime(g.ts.iloc[0])

            if metric == "rainfall" and callable(apply_rain_event_rules):
                apply_rain_event_rules(window_df, t_cur)

            elif metric == "wind_speed" and callable(apply_wind_speed_rules):
                apply_wind_speed_rules(window_df, t_cur)

            elif metric == "temperature" and callable(apply_temp_tod_rules):
                # Build baseline if needed, attach z_tod/resid, then run temp rule
                if self._ensure_temp_baseline() and self._temp_tod_ref is not None:
                    df_t = attach_tod_residuals(window_df, self._temp_tod_ref)
                    apply_temp_tod_rules(df_t, t_cur,
                                        z_hi=settings.temp_z_tod_hi,
                                        z_lo=settings.temp_z_tod_lo,
                                        need=settings.simple_sustained_ticks.get("temperature", 2),
                                        delta_min=settings.temp_delta_min_abs)
                else:
                    # fallback if baseline not ready yet
                    apply_simple_delta_rules(window_df, metric, t_cur)

            else:
                # fallback simple delta rules for everything else
                apply_simple_delta_rules(window_df, metric, t_cur)

            scored_metrics += 1

        logger.info("Scored {} stations across {} metrics", len(latest), scored_metrics)

    def consumer(self):
        while True:
            ts = self.q.get()
            try:
                self.process_tick(ts)
            except Exception as e:
                logger.error(f"engine error: {e}")
            finally:
                self.q.task_done()
