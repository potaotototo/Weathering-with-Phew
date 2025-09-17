import signal
import time
from datetime import datetime, timezone

from .collector import poll_once
from .engine import Engine
from .stations import station_index
from .config import settings
from .store import init_db
from .log import logger

_STOP = False

def _handle_signal(sig, frame):
    global _STOP
    _STOP = True
    logger.info("Shutting down daemon...")

def run_daemon():
    """
    Simple 1-process loop:
      poll_once() -> KDTree refresh (periodic) -> Engine.process_tick()
    Sleeps settings.poll_interval_seconds between iterations.
    """
    init_db() # ensure schema (idempotent)
    eng = Engine()
    station_index._hydrate() # initial neighbor graph

    tick = 0
    backoff = 1 # seconds, exponential on errors up to 60s

    logger.info(
        "Daemon started: poll every {}s, neighbor_k={}, window={}m",
        settings.poll_interval_seconds, settings.neighbor_k, settings.rolling_window_minutes
    )

    while not _STOP:
        try:
            ingested = poll_once()

            # refresh neighbor KDTree periodically (in case new stations appear)
            if tick % 10 == 0:
                station_index._hydrate()

            eng.process_tick(datetime.now(timezone.utc))

            logger.info("Loop ok: ingested={}, tick={}", ingested, tick)
            tick += 1
            backoff = 1
            time.sleep(settings.poll_interval_seconds)

        except Exception as e:
            logger.exception("Daemon iteration failed")
            time.sleep(min(60, backoff))
            backoff = min(60, backoff * 2)

    logger.info("Daemon stopped cleanly.")

if __name__ == "__main__":
    # install signal handlers only when executed directly
    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        pass  # Windows or restricted envs
    run_daemon()
