import pandas as pd, matplotlib.pyplot as plt, os
IN = "data/yolo_out/minute_counts.csv"
df = pd.read_csv(IN, parse_dates=["ts_minute"]).set_index("ts_minute")
print(df.head(10))

# 1) total per minute
plt.figure()
df["total_vehicles"].plot(title="Vehicles per minute")
plt.tight_layout(); os.makedirs("data/plots", exist_ok=True)
plt.savefig("data/plots/minute_total.png")

# 2) hourly average profile
hr = df["total_vehicles"].groupby(df.index.hour).mean()
plt.figure()
hr.plot(kind="bar", title="Avg vehicles by hour")
plt.tight_layout(); plt.savefig("data/plots/hour_profile.png")

# 3) weekday profile
wd = df["total_vehicles"].groupby(df.index.dayofweek).mean()
plt.figure()
wd.plot(kind="bar", title="Avg vehicles by weekday (0=Mon)")
plt.tight_layout(); plt.savefig("data/plots/weekday_profile.png")

print("[DONE] saved plots in data/plots/")
