#!/usr/bin/env python3
from pathlib import Path
import time, json, csv, hashlib, logging as log, datetime as dt, mimetypes
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Paths
BASE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE / "config.json"
LOG_DIR = BASE / "data" / "logs"
RAW_ROOT = BASE / "data" / "raw" / "images"
IMAGES_LOG = LOG_DIR / "images_log.csv"

def write_csv_row(path, headers, row):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    newfile = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if newfile:
            w.writeheader()
        w.writerow(row)

def find_snapshot_url(html, selector):
    try:
        soup = BeautifulSoup(html, "html.parser")
        node = soup.select_one(selector)
        if not node:
            return None
        if node.has_attr("src"):
            return node["src"]
        if node.has_attr("data-src"):
            return node["data-src"]
    except Exception as e:
        log.warning("find_snapshot_url exception: %s", e)
    return None

def _guess_ext_from_content_type(header_value: str):
    if not header_value:
        return ".gif"  # BCN gifs by default
    ctype = header_value.split(";", 1)[0].strip().lower()
    ext = mimetypes.guess_extension(ctype) if ctype else None
    if not ext:
        if ctype == "image/gif":  ext = ".gif"
        elif ctype == "image/png": ext = ".png"
        elif ctype == "image/jpeg": ext = ".jpg"
        else: ext = ".gif"
    return ext

def main(config_path: str | None = None):
    """Run one capture window."""
    if config_path is None:
        config_path = str(DEFAULT_CONFIG)

    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        log.error("Failed to load config %s: %s", config_path, e)
        return False

    cadence = int(cfg.get("cadence_sec", 310))
    window_minutes = int(cfg.get("window_minutes", 20))
    timeout = int(cfg.get("timeout_sec", 12))
    retry_delay = int(cfg.get("retry_delay_sec", 5))
    ua = {"User-Agent": cfg.get("user_agent", "Mozilla/5.0")}

    page_url = cfg.get("camera_page_url", "")
    selector = (cfg.get("image_css_selector") or "").strip()
    direct_url = cfg.get("direct_image_url") or page_url

    start_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    end_utc = start_utc + dt.timedelta(minutes=window_minutes)
    log.info("Window started at %s, will end at %s", start_utc.isoformat(), end_utc.isoformat())

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    while dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc) < end_utc:
        loop_ts = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

        # 1) Try selector first
        img_url = None
        if selector and page_url:
            try:
                r = requests.get(page_url, headers=ua, timeout=timeout)
                r.raise_for_status()
                found = find_snapshot_url(r.text, selector)
                if found:
                    img_url = urljoin(page_url, found)
            except Exception as e:
                log.warning("Selector fetch failed: %s", e)

        # 2) Fallback to direct
        if not img_url:
            img_url = direct_url

        # Download image
        try:
            r = requests.get(img_url, headers=ua, timeout=timeout)
            r.raise_for_status()
            content = r.content
            content_type = r.headers.get("Content-Type", "")
            ext = _guess_ext_from_content_type(content_type)
        except Exception as e:
            log.error("Image download failed for %s: %s", img_url, e)
            write_csv_row(
                IMAGES_LOG,
                ["utc_ts","local_ts","file_path","bytes","sha1","snapshot_url","error"],
                {
                    "utc_ts": loop_ts.isoformat(),
                    "local_ts": dt.datetime.now().astimezone().isoformat(),
                    "file_path": "",
                    "bytes": 0,
                    "sha1": "",
                    "snapshot_url": img_url,
                    "error": str(e),
                },
            )
            time.sleep(retry_delay)
            continue

        # Save file
        day_dir = RAW_ROOT / loop_ts.strftime("%Y/%m/%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        fname = loop_ts.strftime("%Y%m%d_%H%M%S") + ext
        path = day_dir / fname
        try:
            path.write_bytes(content)
        except Exception as e:
            log.error("Failed to write file %s: %s", path, e)
            write_csv_row(
                IMAGES_LOG,
                ["utc_ts","local_ts","file_path","bytes","sha1","snapshot_url","error"],
                {
                    "utc_ts": loop_ts.isoformat(),
                    "local_ts": dt.datetime.now().astimezone().isoformat(),
                    "file_path": str(path.resolve()),
                    "bytes": 0,
                    "sha1": "",
                    "snapshot_url": img_url,
                    "error": f"write failed: {e}",
                },
            )
            time.sleep(retry_delay)
            continue

        sha1 = hashlib.sha1(content).hexdigest()
        write_csv_row(
            IMAGES_LOG,
            ["utc_ts","local_ts","file_path","bytes","sha1","snapshot_url","error"],
            {
                "utc_ts": loop_ts.isoformat(),
                "local_ts": dt.datetime.now().astimezone().isoformat(),
                "file_path": str(path.resolve()),
                "bytes": len(content),
                "sha1": sha1,
                "snapshot_url": img_url,
                "error": "",
            },
        )
        log.info("Saved %s (%d bytes)", path, len(content))
        time.sleep(cadence)

    log.info("Window finished.")
    return True

def run_window(config_path: str | None = None):
    """Entry point used by the scheduler."""
    return main(config_path)

if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()

