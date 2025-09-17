import signal
import time
from datetime import datetime, timezone

from app.collector import poll_once
from app.engine import Engine
from app.stations import station_index
from app.config import settings
from app.log import logger

_STOP = False

def _handle_signal(sig, frame):
    global _STOP
    _STOP = True
    logger.info("Shutting down daemon...")

# graceful shutdown on Ctrl+C / kill
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

def run_daemon():
    """
    Simple 1-process loop:
      poll_once() -> KDTree refresh (periodic) -> Engine.process_tick()
    Sleeps settings.poll_interval_seconds between iterations.
    """
    eng = Engine()
    station_index._hydrate()  # initial neighbor graph

    tick = 0
    backoff = 1  # seconds, exponential on errors up to 60s

    logger.info(
        "Daemon started: poll every {}s, neighbor_k={}, window={}m",
        settings.poll_interval_seconds, settings.neighbor_k, settings.rolling_window_minutes
    )

    while not _STOP:
        try:
            # 1) Ingest latest for all configured metrics
            ingested = poll_once()

            # 2) Refresh neighbor KDTree periodically (new stations appear over time)
            if tick % 10 == 0:  # every ~10 loops
                station_index._hydrate()

            # 3) Score + rules
            eng.process_tick(datetime.now(timezone.utc))

            logger.info("Loop ok: ingested={}, tick={}", ingested, tick)
            tick += 1
            backoff = 1
            time.sleep(settings.poll_interval_seconds)

        except Exception as e:
            logger.error("Daemon iteration failed: {}", e)
            time.sleep(min(60, backoff))
            backoff = min(60, backoff * 2)

    logger.info("Daemon stopped cleanly.")

if __name__ == "__main__":
    run_daemon()
