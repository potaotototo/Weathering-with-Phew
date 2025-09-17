import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.neighbors import KDTree

from .store import upsert_stations, all_stations
from .log import logger


def _to_xy_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Approximate lat/lon -> local planar (x,y) in kilometers.
    Good enough for SG scale (<60km across).
    """
    Rkm = 111.32  # km per degree latitude
    lat0 = np.nanmean(lats) if lats.size else 0.0
    x = (lons - np.nanmean(lons)) * np.cos(np.deg2rad(lat0)) * Rkm
    y = (lats - np.nanmean(lats)) * Rkm
    return np.c_[x, y]


class StationIndex:
    """
    KDTree over station positions (in km) for fast nearest-neighbor lookups.
    Exposes:
      - ready()
      - count
      - neighbors(station_id, k) -> List[(neighbor_id, distance_km)]
      - name(station_id) -> str
      - info(station_id) -> dict
      - _hydrate() to rebuild from DB
      - load_from_file(path) to seed DB and hydrate
    """
    def __init__(self):
        self._stations: List[Tuple[str, str, float, float]] = []   # (id, name, lat, lon)
        self._id_to_idx: Dict[str, int] = {}
        self._tree: Optional[KDTree] = None
        self._coords_km: Optional[np.ndarray] = None

    # ----------- building ------------

    def load_from_file(self, path: str):
        if not os.path.exists(path):
            logger.warning("Station file {} not found; you can populate via collector on first pull.", path)
            return
        with open(path, "r") as f:
            data = json.load(f)
        rows = []
        for s in data:
            try:
                rows.append((s["id"], s.get("name", s["id"]), float(s["lat"]), float(s["lon"])))
            except Exception:
                continue
        if rows:
            upsert_stations(rows)
        self._hydrate()

    def _hydrate(self):
        self._stations = all_stations()
        if not self._stations:
            self._tree = None
            self._coords_km = None
            self._id_to_idx = {}
            return

        lats = np.array([r[2] for r in self._stations], dtype=float)
        lons = np.array([r[3] for r in self._stations], dtype=float)
        self._coords_km = _to_xy_km(lats, lons)

        # Build KDTree in km so returned distances are in kilometers
        self._tree = KDTree(self._coords_km, metric="euclidean")
        self._id_to_idx = {r[0]: i for i, r in enumerate(self._stations)}
        logger.info("KDTree built for {} stations", len(self._stations))

    # ----------- queries ------------

    def ready(self) -> bool:
        return (self._tree is not None) and (self._coords_km is not None) and (len(self._stations) >= 2)

    @property
    def count(self) -> int:
        return len(self._stations)

    def neighbors(self, station_id: str, k: int) -> List[Tuple[str, float]]:
        """
        Return up to k nearest neighbor stations as [(station_id, distance_km)].
        Excludes the station itself. Safe if k exceeds available neighbors.
        """
        if not self.ready() or station_id not in self._id_to_idx:
            return []
        idx = self._id_to_idx[station_id]
        k_eff = max(1, min(k + 1, len(self._stations)))  # +1 to include self, then skip
        dist, ind = self._tree.query(self._coords_km[idx][None, :], k=k_eff)
        out: List[Tuple[str, float]] = []
        for j, d in zip(ind[0], dist[0]):
            sid = self._stations[j][0]
            if sid == station_id:
                continue
            out.append((sid, float(d)))  # d is already in km
        # ensure at most k
        return out[:k]

    def name(self, station_id: str) -> str:
        """Return human-readable name or the id if unknown."""
        i = self._id_to_idx.get(station_id)
        if i is None:
            return station_id
        return self._stations[i][1] or station_id

    def info(self, station_id: str) -> Dict:
        """Return dict with id, name, lat, lon (or empty dict if unknown)."""
        i = self._id_to_idx.get(station_id)
        if i is None:
            return {}
        sid, name, lat, lon = self._stations[i]
        return {"station_id": sid, "name": name, "lat": lat, "lon": lon}


station_index = StationIndex()
