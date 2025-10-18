import os, sys, pandas as pd
import numpy as np

IN = sys.argv[1] if len(sys.argv) > 1 else "data/yolo_out/detections.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/yolo_out/minute_counts.csv"

VEH_SET = {"car","truck","bus","motorcycle","motorbike","bicycle","van"}

def wide_mode(df):
    """Return True if file is in wide format (one row per image, count columns per class)."""
    cols = set(c.lower() for c in df.columns)
    return "utc_ts" in cols and any(c in cols for c in VEH_SET)

def parse_ts_from_path(p):
    """Fallback: parse YYYYMMDD_HHMMSS from filename."""
    import re, datetime as dt, os
    b = os.path.basename(str(p))
    m = re.search(r"(\d{8}_\d{6})", b)
    if not m:
        return pd.NaT
    try:
        return pd.to_datetime(m.group(1), format="%Y%m%d_%H%M%S", utc=True)
    except Exception:
        return pd.NaT

def main():
    df = pd.read_csv(IN)
    print(f"[INFO] Loaded {len(df):,} rows from {IN}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Normalize column names (lower)
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c: c.lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    if wide_mode(df):
        # ---------- WIDE FORMAT ----------
        print("[INFO] Detected WIDE format (counts per class per image).")
        # Time column
        if "utc_ts" in df.columns:
            df["ts"] = pd.to_datetime(df["utc_ts"], utc=True, errors="coerce")
        else:
            src_col = "file_path" if "file_path" in df.columns else "path"
            df["ts"] = df[src_col].map(parse_ts_from_path) if src_col in df.columns else pd.NaT

        df = df.dropna(subset=["ts"]).copy()
        df["ts_minute"] = df["ts"].dt.floor("min")

        # Vehicle columns that actually exist
        veh_cols = [c for c in df.columns if c in VEH_SET]
        if not veh_cols:
            raise ValueError("No vehicle count columns found in wide CSV. Looked for: " + str(sorted(VEH_SET)))

        # Sum counts per minute
        per_min = (df.groupby("ts_minute")[veh_cols]
                     .sum()
                     .sort_index())

        # total_vehicles: if provided use it; else compute
        if "total_vehicles" in df.columns:
            tot = df.groupby("ts_minute")["total_vehicles"].sum()
            per_min["total_vehicles"] = tot
        else:
            per_min["total_vehicles"] = per_min.sum(axis=1)

    else:
        # ---------- LONG/DETECTION FORMAT ----------
        print("[INFO] Detected LONG format (one row per detection).")
        # Time column
        if "ts" not in df.columns:
            src_col = "file_path" if "file_path" in df.columns else "path"
            if src_col not in df.columns:
                raise ValueError("Need either 'ts' or a 'file_path/path' column to parse timestamps.")
            df["ts"] = df[src_col].map(parse_ts_from_path)

        df = df.dropna(subset=["ts"]).copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).copy()
        df["ts_minute"] = df["ts"].dt.floor("min")

        # class column autodetect
        class_cols = [c for c in df.columns if c.lower() in {"cls","class","class_name","name","label"}]
        if not class_cols:
            raise ValueError("Could not find class column among: 'cls','class','class_name','name','label'")
        cls_col = class_cols[0]
        df[cls_col] = (df[cls_col].astype(str).str.lower().str.replace(" ", "", regex=False))

        veh = df[df[cls_col].isin(VEH_SET)].copy()
        if veh.empty:
            print("[WARN] No vehicle detections found.")
            return

        per_min = (veh.groupby(["ts_minute", cls_col])
                     .size()
                     .unstack(fill_value=0)
                     .sort_index())

        # ensure all vehicle columns exist
        for v in VEH_SET:
            if v not in per_min.columns:
                per_min[v] = 0
        per_min = per_min[sorted([c for c in per_min.columns if c in VEH_SET])]
        per_min["total_vehicles"] = per_min.sum(axis=1)

    # ---------- Add features ----------
    idx = per_min.index
    out = per_min.copy()
    out["hour"] = idx.hour
    out["minute"] = idx.minute
    out["dow"] = idx.dayofweek
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["lag_1"] = out["total_vehicles"].shift(1)
    out["lag_3"] = out["total_vehicles"].shift(3)
    out["lag_6"] = out["total_vehicles"].shift(6)
    out = out.dropna()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index_label="ts_minute")
    print(f"[DONE] Wrote {OUT} with shape {out.shape}")

if __name__ == "__main__":
    main()
