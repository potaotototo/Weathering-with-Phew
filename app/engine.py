from queue import Queue
from threading import Thread
from datetime import datetime
import pandas as pd
import json
from .config import settings
from .store import get_latest_readings, write_scores
from .stations import station_index
from .features import compute_univariate_feats, neighbor_gap
from .model import IsoForestModel
from .rules import apply_simple_delta_rules
from .log import logger

class Engine:
    def __init__(self):
        self.q = Queue(maxsize=100)
        self.model = IsoForestModel()

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
        cadence_min = 5                      # NEA realtime cadence
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

            # Train on historical feats (neighbor_gap not required)
            self.model.fit_if_needed(feat_m)

            g = g.copy()
            # Score NEEDS neighbor_gap -> present in 'g'
            g["score"] = self.model.score(g)

            # Neighbor coherence from this timestamp snapshot
            idx = g.set_index("station_id")
            nb_flag, nb_count = [], []
            for sid in g.station_id:
                nb_ids = [n for n,_ in neighbors_map.get(sid, [])]
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

            window_df = feat_m[["ts", "station_id", "metric", "value"]].copy()
            apply_simple_delta_rules(window_df, metric, pd.to_datetime(g.ts.iloc[0]))

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
