import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DET = BASE / "data/yolo_out/detections.csv"
OUT_DS = BASE / "data/datasets"
OUT_DOCS = BASE / "docs/outputs"
OUT_DS.mkdir(parents=True, exist_ok=True)
OUT_DOCS.mkdir(parents=True, exist_ok=True)

def load_detections(det_path: Path) -> pd.DataFrame:
    if not det_path.exists():
        raise FileNotFoundError(f"Missing detections file: {det_path}")
    df = pd.read_csv(det_path)
    # Normalize timestamp column
    ts_col = None
    for c in ["utc_ts","ts","timestamp","utc_ts_iso"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"Could not find timestamp column in {det_path}. Columns: {list(df.columns)}")
    df.rename(columns={ts_col: "utc_ts"}, inplace=True)
    df["utc_ts"] = pd.to_datetime(df["utc_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["utc_ts"]).copy()

    # If YOLO exported long format (one row per detection), pivot into wide counts
    # We detect that by presence of a 'name' or 'class' column without per-class columns.
    per_class_cols = {"car","truck","bus","motorcycle","bicycle","person"}
    has_wide = any(c in df.columns for c in per_class_cols)
    if not has_wide:
        # Expect columns like: file_path, name (class), confidence, etc.
        name_col = None
        for c in ["name","class","cls_name","label"]:
            if c in df.columns:
                name_col = c
                break
        if name_col is None:
            raise ValueError("Detections are not in wide format and no class name column found.")
        # Make counts per (utc_ts, file_path, name)
        base_cols = [c for c in ["utc_ts","file_path"] if c in df.columns]
        grp = (df.groupby(base_cols + [name_col])
                 .size()
                 .reset_index(name="count"))
        wide = grp.pivot_table(index=base_cols, columns=name_col, values="count", aggfunc="sum", fill_value=0)
        wide.columns = [str(c).lower().replace(" ", "") for c in wide.columns]
        wide = wide.reset_index()
        # Ensure standard class columns exist
        for need in per_class_cols:
            if need not in wide.columns:
                wide[need] = 0
        df = wide

    # Ensure all expected count columns exist
    for c in ["car","truck","bus","motorcycle","bicycle","person"]:
        if c not in df.columns:
            df[c] = 0

    # Total vehicles (exclude person)
    if "total_vehicles" not in df.columns:
        df["total_vehicles"] = df[["car","truck","bus","motorcycle","bicycle"]].sum(axis=1)

    # Keep essential columns
    keep = ["utc_ts","file_path","car","truck","bus","motorcycle","bicycle","person","total_vehicles"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Add convenience time columns (UTC-based)
    df["hour_utc"] = df["utc_ts"].dt.hour
    df["minute_utc"] = df["utc_ts"].dt.minute
    df["dow_utc"] = df["utc_ts"].dt.day_name()
    df["is_weekend"] = df["dow_utc"].isin(["Saturday","Sunday"]).astype(int)
    return df

def per_minute_aggregate(per_image: pd.DataFrame) -> pd.DataFrame:
    d = per_image.copy()
    d["ts_minute"] = d["utc_ts"].dt.floor("min")
    agg_cols = ["car","truck","bus","motorcycle","bicycle","person","total_vehicles"]
    g = (d.groupby("ts_minute")[agg_cols]
           .sum()
           .reset_index())
    # Add features
    g["hour"] = g["ts_minute"].dt.hour
    g["minute"] = g["ts_minute"].dt.minute
    g["dow"] = g["ts_minute"].dt.day_name()
    g["is_weekend"] = g["dow"].isin(["Saturday","Sunday"]).astype(int)
    # Lags on total_vehicles
    g = g.sort_values("ts_minute")
    g["lag_1"] = g["total_vehicles"].shift(1)
    g["lag_3"] = g["total_vehicles"].shift(3)
    g["lag_6"] = g["total_vehicles"].shift(6)
    return g

def per_hour_aggregate(per_min: pd.DataFrame) -> pd.DataFrame:
    d = per_min.set_index("ts_minute")
    h = (d["total_vehicles"]
            .resample("H")
            .sum()
            .rename("total_vehicles")
            .to_frame())
    h = h.reset_index().rename(columns={"ts_minute":"ts_hour_utc"})
    return h

def write_qc(per_image: pd.DataFrame, per_min: pd.DataFrame, per_hour: pd.DataFrame, out_txt: Path):
    lines = []
    lines.append("=== DATASET QUALITY REPORT ===")
    lines.append("")
    # Image-level
    lines.append(f"Images (rows in per_image_clean.csv): {len(per_image):,}")
    tmin = per_image['utc_ts'].min()
    tmax = per_image['utc_ts'].max()
    lines.append(f"UTC range: {tmin} → {tmax}")
    # Minute coverage
    all_minutes = pd.date_range(start=per_min['ts_minute'].min(),
                                end=per_min['ts_minute'].max(),
                                freq="min")
    coverage = 100.0 * len(per_min) / max(1, len(all_minutes))
    lines.append(f"Minute rows: {len(per_min):,} (coverage vs full range: {coverage:.1f}%)")
    # Busiest minutes
    top = per_min.nlargest(5, "total_vehicles")[["ts_minute","total_vehicles"]]
    lines.append("")
    lines.append("Top 5 busiest minutes:")
    for _, r in top.iterrows():
        lines.append(f"  {r['ts_minute']}  ->  {int(r['total_vehicles'])} vehicles")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

def make_plots(per_min: pd.DataFrame, per_hour: pd.DataFrame, out_dir: Path):
    # Hourly total line
    plt.figure()
    plt.plot(per_hour["ts_hour_utc"], per_hour["total_vehicles"])
    plt.title("Vehicles per Hour (UTC)")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Vehicles/hour")
    plt.tight_layout()
    (out_dir / "plot_hourly_total.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "plot_hourly_total.png")
    plt.close()

    # Weekday bar (vehicles per minute avg)
    tmp = per_min.copy()
    tmp["dow"] = tmp["ts_minute"].dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    bar = (tmp.groupby("dow")["total_vehicles"].mean()
             .reindex(order))
    plt.figure()
    bar.plot(kind="bar")
    plt.title("Average Vehicles per Minute by Weekday (UTC)")
    plt.xlabel("Day of week")
    plt.ylabel("Vehicles/min")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_weekday_bar.png")
    plt.close()

def main():
    per_image = load_detections(DET)
    per_image_out = OUT_DS / "per_image_clean.csv"
    per_image.to_csv(per_image_out, index=False)
    print(f"[DONE] {per_image_out}")

    per_min = per_minute_aggregate(per_image)
    per_min_out = OUT_DS / "per_minute.csv"
    per_min.to_csv(per_min_out, index=False)
    print(f"[DONE] {per_min_out}")

    per_hour = per_hour_aggregate(per_min)
    per_hour_out = OUT_DS / "per_hour.csv"
    per_hour.to_csv(per_hour_out, index=False)
    print(f"[DONE] {per_hour_out}")

    qc_out = OUT_DOCS / "dataset_qc.txt"
    write_qc(per_image, per_min, per_hour, qc_out)
    print(f"[DONE] {qc_out}")

    make_plots(per_min, per_hour, OUT_DOCS)
    print(f"[DONE] {OUT_DOCS / 'plot_hourly_total.png'}")
    print(f"[DONE] {OUT_DOCS / 'plot_weekday_bar.png'}")
    print("[ALL GOOD] Build dataset complete.")

if __name__ == "__main__":
    main()
