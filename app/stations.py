import json
import os
import numpy as np
from sklearn.neighbors import KDTree
from typing import Dict, List, Tuple
from .store import upsert_stations, all_stations
from .log import logger

class StationIndex:
    def __init__(self):
        self._stations = []  # (id, name, lat, lon)
        self._id_to_idx = {}
        self._tree = None

    def load_from_file(self, path: str):
        if not os.path.exists(path):
            logger.warning("Station file {} not found; you can populate via collector on first pull.", path)
            return
        with open(path, "r") as f:
            data = json.load(f)
        rows = []
        for s in data:
            rows.append((s["id"], s.get("name", s["id"]), float(s["lat"]), float(s["lon"])) )
        upsert_stations(rows)
        self._hydrate()

    def _hydrate(self):
        self._stations = all_stations()
        if not self._stations:
            return
        coords = np.array([[r[2], r[3]] for r in self._stations])
        self._tree = KDTree(coords, metric='euclidean')
        self._id_to_idx = {r[0]: i for i, r in enumerate(self._stations)}
        logger.info("KDTree built for {} stations", len(self._stations))

    def ready(self):
        return self._tree is not None

    def neighbors(self, station_id: str, k: int) -> List[Tuple[str, float]]:
        if not self.ready() or station_id not in self._id_to_idx:
            return []
        idx = self._id_to_idx[station_id]
        dist, ind = self._tree.query([self._tree.data[idx]], k=k+1)  # include self
        out = []
        for j, d in zip(ind[0], dist[0]):
            sid = self._stations[j][0]
            if sid == station_id:  # skip self
                continue
            out.append((sid, float(d)))
        return out

station_index = StationIndex()
