"""
Model Connector — bridges user inputs (from NLP or direct calls) to the trained
Random Forest classifier, handling weather fetching and lag defaults automatically.

Usage:
    from src.connector import ModelConnector

    mc = ModelConnector()
    result = mc.predict(hour=8, day_of_week=0)
    print(result)
    # {
    #   "traffic_level": "High",
    #   "confidence": 0.82,
    #   "probabilities": {"High": 0.82, "Low": 0.09, "Medium": 0.09},
    #   "day_name": "Monday",
    #   "weather": {"temperature": 18.5, "precipitation": 0.0, "windspeed": 9.2},
    #   "weather_source": "api"   # or "dataset_median" / "fallback"
    # }
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent
_PROJECT_DIR = _SRC_DIR.parent
MODEL_PATH = _PROJECT_DIR / "model_random_forest.pkl"
DATA_PATH = _PROJECT_DIR / "TRAFFIC_ENHANCED.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BCN_LAT = 41.3913
BCN_LON = 2.1649

FEATURE_COLUMNS = [
    "hour", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos",
    "temperature", "precipitation", "windspeed",
    "lag_1", "lag_2", "lag_3",
]

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Main connector class
# ---------------------------------------------------------------------------

class ModelConnector:
    """
    Loads the trained Random Forest classifier and exposes a predict() method.
    Instantiate once and reuse across requests.
    """

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run src/train_random_forest.py first."
            )
        self.model = joblib.load(MODEL_PATH)
        self._lag_defaults: dict[tuple[int, int], tuple[float, float, float]] = {}
        self._weather_defaults: dict[int, dict[str, float]] = {}
        self._load_dataset_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        hour: int,
        day_of_week: int,
        temperature: Optional[float] = None,
        precipitation: Optional[float] = None,
        windspeed: Optional[float] = None,
        lag_1: Optional[float] = None,
        lag_2: Optional[float] = None,
        lag_3: Optional[float] = None,
    ) -> dict:
        """
        Predict traffic level for a given hour and day.

        Args:
            hour:        0–23  (local Barcelona time)
            day_of_week: 0=Monday … 6=Sunday
            temperature: °C override (fetched from Open-Meteo if omitted)
            precipitation: mm override
            windspeed:   km/h override
            lag_1/2/3:   previous vehicle counts (dataset medians used if omitted)

        Returns dict with keys:
            traffic_level   – "Low" | "Medium" | "High"
            confidence      – probability of the predicted class (0–1)
            probabilities   – full class probability dict
            day_name        – human-readable day name
            weather         – resolved weather dict
            weather_source  – "api" | "dataset_median" | "fallback"
        """
        if not (0 <= hour <= 23):
            raise ValueError(f"hour must be 0–23, got {hour}")
        if not (0 <= day_of_week <= 6):
            raise ValueError(f"day_of_week must be 0–6, got {day_of_week}")

        weather, weather_source = self._resolve_weather(
            hour, temperature, precipitation, windspeed
        )
        lag1, lag2, lag3 = self._resolve_lags(hour, day_of_week, lag_1, lag_2, lag_3)

        is_weekend = 1 if day_of_week >= 5 else 0
        features = pd.DataFrame([{
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "temperature": weather["temperature"],
            "precipitation": weather["precipitation"],
            "windspeed": weather["windspeed"],
            "lag_1": lag1,
            "lag_2": lag2,
            "lag_3": lag3,
        }])[FEATURE_COLUMNS]

        predicted_class = self.model.predict(features)[0]

        probabilities: dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            for cls, p in zip(self.model.classes_, proba):
                p_val: float = float(p)
                probabilities[str(cls)] = round(p_val, 4)
        conf_raw: float = float(probabilities.get(predicted_class, 1.0))

        return {
            "traffic_level": predicted_class,
            "confidence": round(conf_raw, 4),
            "probabilities": probabilities,
            "day_name": DAY_NAMES[day_of_week],
            "weather": weather,
            "weather_source": weather_source,
        }

    # ------------------------------------------------------------------
    # Feature resolution helpers
    # ------------------------------------------------------------------

    def _resolve_weather(
        self,
        hour: int,
        temperature: Optional[float],
        precipitation: Optional[float],
        windspeed: Optional[float],
    ) -> tuple[dict, str]:
        """Return (weather_dict, source_label). Fetch API if values are missing."""
        if temperature is not None and precipitation is not None and windspeed is not None:
            return (
                {
                    "temperature": float(temperature),
                    "precipitation": float(precipitation),
                    "windspeed": float(windspeed),
                },
                "override",
            )

        fetched, source = _fetch_weather_for_hour(hour)

        return (
            {
                "temperature": temperature if temperature is not None
                               else fetched.get("temperature", self._weather_default(hour, "temperature")),
                "precipitation": precipitation if precipitation is not None
                                 else fetched.get("precipitation", self._weather_default(hour, "precipitation")),
                "windspeed": windspeed if windspeed is not None
                             else fetched.get("windspeed", self._weather_default(hour, "windspeed")),
            },
            source,
        )

    def _resolve_lags(
        self,
        hour: int,
        day_of_week: int,
        lag_1: Optional[float],
        lag_2: Optional[float],
        lag_3: Optional[float],
    ) -> tuple[float, float, float]:
        """Fill missing lags from per-(hour, day_of_week) dataset medians."""
        d1, d2, d3 = self._lag_defaults.get((hour, day_of_week), (3.0, 3.0, 3.0))
        return (
            float(lag_1) if lag_1 is not None else d1,
            float(lag_2) if lag_2 is not None else d2,
            float(lag_3) if lag_3 is not None else d3,
        )

    def _weather_default(self, hour: int, key: str) -> float:
        return self._weather_defaults.get(hour, {}).get(
            key, {"temperature": 20.0, "precipitation": 0.0, "windspeed": 8.0}[key]
        )

    # ------------------------------------------------------------------
    # Dataset defaults pre-computation
    # ------------------------------------------------------------------

    def _load_dataset_defaults(self) -> None:
        """Pre-compute median lag and weather values per (hour, day_of_week)."""
        if not DATA_PATH.exists():
            return
        try:
            df = pd.read_csv(DATA_PATH)
            for (h, d), group in df.groupby(["hour", "day_of_week"]):
                self._lag_defaults[(int(h), int(d))] = (
                    float(group["lag_1"].median()),
                    float(group["lag_2"].median()),
                    float(group["lag_3"].median()),
                )
            for h, group in df.groupby("hour"):
                self._weather_defaults[int(h)] = {
                    "temperature": float(group["temperature"].median()),
                    "precipitation": float(group["precipitation"].median()),
                    "windspeed": float(group["windspeed"].median()),
                }
        except Exception:
            pass  # defaults remain empty; _resolve_* will use hardcoded fallbacks


# ---------------------------------------------------------------------------
# Open-Meteo weather fetch
# ---------------------------------------------------------------------------

def _fetch_weather_for_hour(target_hour: int) -> tuple[dict, str]:
    """
    Fetch hourly forecast for Barcelona from Open-Meteo.
    Returns (weather_dict, source) where source is "api" or "fallback".
    Picks the next upcoming occurrence of target_hour in the 48-hour forecast.
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={BCN_LAT}&longitude={BCN_LON}"
            "&hourly=temperature_2m,precipitation,windspeed_10m"
            "&timezone=Europe%2FMadrid"
            "&forecast_days=2"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        times = data["hourly"]["time"]           # ["2025-10-10T00:00", ...]
        temps = data["hourly"]["temperature_2m"]
        precips = data["hourly"]["precipitation"]
        winds = data["hourly"]["windspeed_10m"]

        now_hour = datetime.now(timezone.utc).hour
        best_index = None

        # Prefer the soonest upcoming slot that matches target_hour
        for i, t in enumerate(times):
            slot_hour = int(t[11:13])
            if slot_hour == target_hour:
                if best_index is None:
                    best_index = i
                # prefer a slot whose wall-clock hour is >= now
                elif int(t[8:10]) >= datetime.now().day:
                    best_index = i
                    break

        if best_index is not None:
            t_val: float = float(temps[best_index])
            p_val: float = float(precips[best_index])
            w_val: float = float(winds[best_index])
            return (
                {
                    "temperature": round(t_val, 1),
                    "precipitation": round(p_val, 2),
                    "windspeed": round(w_val, 1),
                },
                "api",
            )
    except Exception:
        pass

    return {}, "fallback"


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mc = ModelConnector()
    print("Dataset lag defaults loaded:", len(mc._lag_defaults), "slots")
    print("Dataset weather defaults loaded:", len(mc._weather_defaults), "hours")
    print()

    test_cases = [
        (8,  0, "Monday morning rush"),
        (14, 2, "Wednesday afternoon"),
        (22, 5, "Saturday night"),
        (3,  6, "Sunday early morning"),
    ]

    for hour, dow, label in test_cases:
        result = mc.predict(hour=hour, day_of_week=dow)
        w = result["weather"]
        print(
            f"{label:30s} -> {result['traffic_level']:6s} "
            f"(conf {result['confidence']*100:.1f}%) "
            f"| {result['day_name']} {hour:02d}:00 "
            f"| {w['temperature']}°C  {w['precipitation']}mm  {w['windspeed']}km/h wind "
            f"[weather: {result['weather_source']}]"
        )
