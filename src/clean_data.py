import pandas as pd

def clean_data():
    print("--- STEP 1: DATA CLEANING ---")
    
    # 1. Load Raw Data
    try:
        df = pd.read_csv('data/yolo_out/minute_counts.csv')
    except FileNotFoundError:
        print("Error: minute_counts.csv not found.")
        return

    # 2. Fix Time (Convert UTC to Barcelona Time)
    df['utc_ts'] = pd.to_datetime(df['utc_ts'])
    df['timestamp'] = df['utc_ts'].dt.tz_convert('Europe/Madrid')

    # 3. Aggregate by Hour (Get average cars per hour)
    df.set_index('timestamp', inplace=True)
    df_hourly = df['total_vehicles'].resample('h').mean().reset_index()
    
    # Drop empty hours
    df_hourly.dropna(inplace=True)

    # 4. Create Features (The Inputs)
    df_hourly['hour'] = df_hourly['timestamp'].dt.hour
    df_hourly['day_of_week'] = df_hourly['timestamp'].dt.dayofweek
    # Weekend: 1 if Saturday(5) or Sunday(6), else 0
    df_hourly['is_weekend'] = df_hourly['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 5. Create Labels for Classification (Low/Medium/High)
    # We use percentiles to decide what is "High" vs "Low"
    low = df_hourly['total_vehicles'].quantile(0.33)
    high = df_hourly['total_vehicles'].quantile(0.66)
    
    def get_level(count):
        if count <= low: return 'Low'
        elif count <= high: return 'Medium'
        else: return 'High'
        
    df_hourly['traffic_level'] = df_hourly['total_vehicles'].apply(get_level)

    # 6. Save
    df_hourly.to_csv('TRAFFIC_CLEANED.csv', index=False)
    print("SUCCESS: Saved 'TRAFFIC_CLEANED.csv'")

if __name__ == "__main__":
    clean_data()
