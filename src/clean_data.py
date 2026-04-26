import pandas as pd
import numpy as np
import requests
from pathlib import Path

# Paths
RAW_CSV = Path("../TRAFFIC_PER_IMAGE.csv")
OUT_PATH = Path("TRAFFIC_ENHANCED.csv")

# Weather API settings
LAT = 41.3913
LON = 2.1649
START_DATE = "2025-10-07"
END_DATE = "2025-11-16"
WEATHER_URL = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={START_DATE}&end_date={END_DATE}&hourly=temperature_2m,precipitation,windspeed_10m&timezone=UTC"

def build_large_dataset():
    print("--- STEP 1 (GRANULAR): BUILDING DATASET FROM IMAGE LOGS ---")
    
    if not RAW_CSV.exists():
        print(f"Error: {RAW_CSV} not found.")
        return

    # 1. Load Image-level Data
    print(f"Loading {RAW_CSV} ({len(pd.read_csv(RAW_CSV))} rows)...")
    df = pd.read_csv(RAW_CSV)
    df['timestamp'] = pd.to_datetime(df['utc_ts'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 2. Fetch Weather Data (UTC)
    print("Fetching hourly weather data for join...")
    try:
        r = requests.get(WEATHER_URL, timeout=15)
        r.raise_for_status()
        w_json = r.json()
        hourly = w_json.get("hourly", {})
        df_weather = pd.DataFrame({
            "ts_hour": pd.to_datetime(hourly.get("time", []), utc=True),
            "temperature": hourly.get("temperature_2m", []),
            "precipitation": hourly.get("precipitation", []),
            "windspeed": hourly.get("windspeed_10m", [])
        })
    except Exception as e:
        print(f"Weather fetch failed: {e}")
        return

    # 3. Merge Weather onto Image-level timestamps
    # We floor our image timestamps to the hour to join with hourly weather
    df['ts_hour'] = df['timestamp'].dt.floor('h')
    print("Merging weather features...")
    df = pd.merge(df, df_weather, on="ts_hour", how="left")

    # 4. Feature Engineering
    print("Adding features (Cyclic Time, Lags)...")
    # Convert to Barcelona time for feature extraction
    df_local = df['timestamp'].dt.tz_convert('Europe/Madrid')
    df['hour'] = df_local.dt.hour
    df['day_of_week'] = df_local.dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Lag Features (Last 3 image-level counts)
    df['lag_1'] = df['total_vehicles'].shift(1)
    df['lag_2'] = df['total_vehicles'].shift(2)
    df['lag_3'] = df['total_vehicles'].shift(3)

    # 5. Define Traffic Levels (based on granular distribution)
    low_thresh = df['total_vehicles'].quantile(0.33)
    high_thresh = df['total_vehicles'].quantile(0.66)
    
    def get_level(val):
        if val <= low_thresh: return 'Low'
        elif val <= high_thresh: return 'Medium'
        else: return 'High'
    
    df['traffic_level'] = df['total_vehicles'].apply(get_level)

    # 6. Cleanup
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    
    cols = [
        'timestamp', 'total_vehicles', 'hour', 'day_of_week', 'is_weekend',
        'hour_sin', 'hour_cos', 'temperature', 'precipitation', 'windspeed',
        'lag_1', 'lag_2', 'lag_3', 'traffic_level'
    ]
    df = df[cols]

    # 7. Save
    df.to_csv(OUT_PATH, index=False)
    print(f"SUCCESS: Enhanced granular dataset saved to '{OUT_PATH}'")
    print(f"Records: {len(df)} (Dropped {initial_len - len(df)} NaNs)")
    print(f"Thresholds used: Low <= {low_thresh:.1f}, High > {high_thresh:.1f}")

if __name__ == "__main__":
    build_large_dataset()
