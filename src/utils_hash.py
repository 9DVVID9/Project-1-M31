
import argparse, os, pathlib, hashlib, csv, datetime as dt

def sha1_of_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedupe", type=str, help="Path to images root folder (e.g., data/raw/images)")
    ap.add_argument("--report_only", action="store_true", help="Only report duplicates; do not delete")
    args = ap.parse_args()

    root = pathlib.Path(args.dedupe).resolve()
    seen = {}
    dups = []

    for p, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".jpg"):
                continue
            fp = pathlib.Path(p) / f
            h = sha1_of_file(fp)
            if h in seen:
                dups.append((h, str(fp), seen[h]))
            else:
                seen[h] = str(fp)

    logdir = root.parents[2] / "logs"  # data/logs
    logdir.mkdir(parents=True, exist_ok=True)
    outcsv = logdir / "duplicates.csv"

    with open(outcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sha1","duplicate_path","kept_path"])
        for h, dup_path, kept in dups:
            w.writerow([h, dup_path, kept])

    print(f"Found {len(dups)} duplicate images. Report saved to {outcsv}")

    if not args.report_only:
        removed = 0
        for _, dup_path, _ in dups:
            try:
                os.remove(dup_path)
                removed += 1
            except Exception as e:
                print("Failed to remove", dup_path, e)
        print(f"Removed {removed} duplicate files.")

if __name__ == "__main__":
    main()
