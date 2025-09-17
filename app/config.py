from pydantic import BaseModel, Field
from typing import Dict, Optional
import os

class Settings(BaseModel):
    # API base + endpoints
    nea_base_url: str = "https://api-open.data.gov.sg/v2/real-time/api"
    nea_endpoints: Dict[str, str] = Field(
        default_factory=lambda: {
            "temperature": "air-temperature",
            "rainfall": "rainfall",
            "humidity": "relative-humidity",
            "wind_direction": "wind-direction",
            "wind_speed": "wind-speed",
        }
    )

    # Core runtime config
    poll_interval_seconds: int = 60
    rolling_window_minutes: int = 90
    neighbor_k: int = 4
    z_threshold: float = 2.0
    sustained_minutes: int = 2
    cooldown_minutes: int = 10
    database_path: str = "./weatherguard.db"
    cache_dir: str = "./data"

    # Neighbor / rule tuning
    weather_neighbor_min: int = 2          # require ≥2 anomalous neighbors for WEATHER
    neighbor_z_threshold: float = 2.0      # neighbor |z| threshold to count as anomalous
    delta_min: Dict[str, float] = Field(   # min absolute per-tick change to pass sustained rule
        default_factory=lambda: {
            "temperature": 0.2,            # °C
            "humidity": 1.0,               # %
            "wind_speed": 0.5,             # kn
            "rainfall": 0.05,              # mm per 5 min
            "wind_direction": 8.0,         # degrees per 5 min
        }
    )
    simple_delta_threshold: Dict[str, float] = Field(
        default_factory=lambda: {
            "temperature":   0.8,   # °C change vs previous tick
            "humidity":      5.0,   # % RH
            "wind_speed":    3.0,   # knots
            "rainfall":      0.2,   # mm in current 5-min reading
            "wind_direction":35.0,  # degrees (circular); was too sensitive before
        }
    )
    simple_sustained_ticks: Dict[str, int] = Field(
        default_factory=lambda: {
            "temperature":   2,     # require N consecutive ticks over threshold
            "humidity":      2,
            "wind_speed":    2,
            "rainfall":      1,     # rain uses current value, 1 tick is fine
            "wind_direction":2,     # helps cut jitter
        }
    )

    # Rain event rules
    rain_onset_min_intervals: int = 2
    rain_onset_min_total_mm: float = 0.2
    rain_intense_total_mm: float = 2.0
    rain_calm_mm: float = 0.05
    rain_stop_quiet_intervals: int = 2
    rain_trend_window: int = 3
    rain_drop_pct_for_easing: float = 0.5
    rain_cooldown_minutes: int = 20

    # Optional API key
    x_api_key: Optional[str] = os.getenv("DATA_GOV_SG_API_KEY")

settings = Settings()
