import pandas as pd, numpy as np, os, sys, datetime as dt

IN = sys.argv[1] if len(sys.argv) > 1 else "data/yolo_out/minute_counts.csv"
HORIZON_MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 24*60  # forecast next 24h
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/yolo_out/forecast_baseline.csv"

df = pd.read_csv(IN, parse_dates=["ts_minute"]).set_index("ts_minute")
s = df["total_vehicles"].copy()

# seasonal mean by (dow,hour,minute)
key = pd.MultiIndex.from_arrays([df.index.dayofweek, df.index.hour, df.index.minute],
                                names=["dow","hour","minute"])
seasonal_mean = s.groupby(key).mean()

# build future index minute-by-minute from last + 1 minute
start = df.index.max() + pd.Timedelta(minutes=1)
future_idx = pd.date_range(start, periods=HORIZON_MIN, freq="min")

# map seasonal means
fkey = pd.MultiIndex.from_arrays([future_idx.dayofweek, future_idx.hour, future_idx.minute],
                                 names=["dow","hour","minute"])
yhat = seasonal_mean.reindex(fkey).values

# fallback if any NaNs (unseen slots): fill with overall mean
overall = s.mean()
yhat = np.where(np.isnan(yhat), overall, yhat)

out = pd.DataFrame({"ts_minute": future_idx, "pred_total_vehicles": yhat})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_csv(OUT, index=False)
print(f"[DONE] wrote {OUT} with {len(out)} rows")
