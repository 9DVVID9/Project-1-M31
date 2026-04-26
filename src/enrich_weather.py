import pandas as pd
import requests
import json
from pathlib import Path

CSV_PATH = Path("TRAFFIC_CLEANED.csv")
# Barcelona Coordinates (Carrer d'Arago central area)
LAT = 41.3913
LON = 2.1649
START_DATE = "2025-10-07"
END_DATE = "2025-11-16"

URL = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={START_DATE}&end_date={END_DATE}&hourly=temperature_2m,precipitation,windspeed_10m&timezone=UTC"

def enrich_traffic_data():
    print(f"--- WEATHER ENRICHMENT ---")
    
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    # 1. Fetch Weather Data
    print(f"Fetching weather data (UTC) for Barcelona ({LAT}, {LON})...")
    try:
        response = requests.get(URL, timeout=15)
        response.raise_for_status()
        weather_data = response.json()
    except Exception as e:
        print(f"Failed to fetch weather data: {e}")
        return

    # 2. Parse Weather to DataFrame
    hourly = weather_data.get("hourly", {})
    df_weather = pd.DataFrame({
        "timestamp_utc": hourly.get("time", []),
        "temperature": hourly.get("temperature_2m", []),
        "precipitation": hourly.get("precipitation", []),
        "windspeed": hourly.get("windspeed_10m", [])
    })
    
    # Convert weather timestamps to UTC datetime objects
    df_weather['timestamp_utc'] = pd.to_datetime(df_weather['timestamp_utc']).dt.tz_localize('UTC')

    # 3. Load Traffic Data
    print(f"Loading {CSV_PATH}...")
    df_traffic = pd.read_csv(CSV_PATH)
    
    # Convert traffic timestamps to UTC for reliable merging
    # Passing utc=True handles mixed timezone strings (like +02:00 and +01:00)
    df_traffic['timestamp_orig'] = df_traffic['timestamp'] 
    df_traffic['timestamp_utc'] = pd.to_datetime(df_traffic['timestamp'], utc=True)

    # 4. Merge Data
    print("Merging weather features...")
    df_merged = pd.merge(df_traffic, df_weather, on="timestamp_utc", how="left")

    # 5. Clean up and Save
    # We want to keep the original timestamp format for consistency
    df_final = df_merged.drop(columns=['timestamp_utc', 'timestamp']).rename(columns={'timestamp_orig': 'timestamp'})
    
    # Reorder columns to put weather near the end
    cols = ['timestamp', 'total_vehicles', 'hour', 'day_of_week', 'is_weekend', 'traffic_level', 'temperature', 'precipitation', 'windspeed']
    df_final = df_final[cols]

    print(f"Saving enriched dataset to {CSV_PATH}...")
    df_final.to_csv(CSV_PATH, index=False)
    print("Done! Weather information added successfully.")

if __name__ == "__main__":
    enrich_traffic_data()
