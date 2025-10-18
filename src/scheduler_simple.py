#!/usr/bin/env python3
import time
import json
import datetime as dt
import importlib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE / "config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_hhmm(s: str):
    parts = s.split(":")
    return int(parts[0]), int(parts[1])

def sleep_until(target: dt.datetime):
    while True:
        now = dt.datetime.now().astimezone()
        delta = (target - now).total_seconds()
        if delta <= 0:
            return
        time.sleep(min(60, max(1, delta)))

def main():
    cfg = load_config()
    run_times = cfg.get("run_times_local", [])
    if not run_times:
        run_times = ["06:00","06:20","06:40","07:00","07:20","07:40",
                     "08:00","08:20","08:40","09:00","09:20","09:40",
                     "10:00","10:20","10:40","11:00","11:20","11:40",
                     "12:00","12:20","12:40","13:00","13:20","13:40",
                     "14:00","14:20","14:40","15:00","15:20","15:40",
                     "16:00","16:20","16:40","17:00","17:20","17:40",
                     "18:00","18:20","18:40","19:00","19:20","19:40",
                     "20:00","20:20","20:40","21:00","21:20","21:40",
                     "22:00","22:20","22:40","23:00","23:20","23:40",
                     "00:00","00:20","00:40","01:00","01:20","01:40",
                     "02:00","02:20","02:40","03:00","03:20","03:40",
                     "04:00","04:20","04:40","05:00","05:20","05:40"]

    total_days = int(cfg.get("total_days", 21))

    grab = importlib.import_module("src.grab_window_bs4")

    days_done = 0
    last_date = None

    print("Scheduler started. Will run for", total_days, "days. Local tz now:",
          dt.datetime.now().astimezone())

    while days_done < total_days:
        now = dt.datetime.now().astimezone()
        today = now.date()
        if last_date is None or today != last_date:
            last_date = today
            print(f"=== Day {days_done+1} of {total_days} ({today}) ===")

        # compute the next upcoming run among the configured run_times
        upcoming = []
        for t in run_times:
            hh, mm = parse_hhmm(t)
            run_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if run_dt <= now:
                run_dt = run_dt + dt.timedelta(days=1)
            upcoming.append(run_dt)

        upcoming.sort()
        target = upcoming[0]
        print("Next window at local time:", target.isoformat())

        sleep_until(target)
        print("Starting window at:", dt.datetime.now().astimezone().isoformat())

        try:
            grab.run_window(str(CONFIG_PATH))
        except Exception as e:
            print("grab.run_window raised:", e)

        print("Window finished at:", dt.datetime.now().astimezone().isoformat())

        now_after = dt.datetime.now().astimezone()
        if now_after.date() != today:
            days_done += 1

    print("All scheduled days completed. Exiting.")

if __name__ == "__main__":
    main()
