#!/usr/bin/env python3
"""
Batch-detect vehicles with YOLOv8 and write a CSV summary.
"""
import argparse, os, sys, csv
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageSequence
from ultralytics import YOLO

INTEREST = ["car","truck","bus","motorcycle","bicycle","person"]

def iter_images(src: Path):
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in (".jpg",".jpeg",".png",".gif"):
            yield p

def load_first_frame(path: Path):
    try:
        with Image.open(path) as im:
            if getattr(im,"is_animated",False):
                im.seek(0)
            return np.array(im.convert("RGB"))
    except Exception as e:
        print(f"[WARN] cannot read {path}: {e}", file=sys.stderr)
        return None

def parse_timestamp_from_name(path: Path):
    stem = path.stem
    try:
        if "_" in stem:
            d, t = stem.split("_")[-2], stem.split("_")[-1]
            if len(d)==8 and len(t)==6:
                return datetime.strptime(d+t,"%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception: pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

def ensure_csv(path: Path, headers):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if new:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/images")
    ap.add_argument("--out", default="data/yolo_out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save-vis", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    vis = out/"annotated"; csv_path = out/"detections.csv"
    out.mkdir(parents=True, exist_ok=True)
    if args.save_vis: vis.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt"); model.to("cpu")

    headers = ["utc_ts","file_path"]+INTEREST+["total_vehicles"]
    ensure_csv(csv_path, headers)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        for i,p in enumerate(iter_images(src),1):
            if args.limit and i>args.limit: break
            arr = load_first_frame(p)
            if arr is None: continue
            r = model.predict(source=arr, imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)
            counts = {k:0 for k in INTEREST}
            if r and r[0].boxes is not None:
                for cid in r[0].boxes.cls.cpu().numpy().astype(int):
                    name = r[0].names.get(cid,"")
                    if name in counts: counts[name]+=1
            tot = sum(counts[k] for k in INTEREST if k!="person")
            w.writerow(dict(utc_ts=parse_timestamp_from_name(p).isoformat(), file_path=str(p), **counts, total_vehicles=tot))
            if args.save_vis:
                try:
                    vis_path = vis/p.relative_to(src)
                    vis_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(r[0].plot()[:,:,::-1]).save(vis_path.with_suffix(".jpg"), quality=90)
                except Exception as e:
                    print(f"[WARN] vis fail {p}: {e}", file=sys.stderr)
            if i%25==0: print(f"[INFO] processed {i} images…")
    print(f"[DONE] results in {csv_path}")

if __name__=="__main__": sys.exit(main())
