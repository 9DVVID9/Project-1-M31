
# Traffic Collection Starter (BeautifulSoup + CSV, no cron)

- **CSV instead of a database** (all logs go to `data/logs/*.csv`).
- **BeautifulSoup for web scraping**
- **20-minute collection windows**, with **images every N seconds**.

> ⚠️ **Legal check**: Only automate downloads if the site/owner allows it (check the page Terms/robots.txt or ask permission). If automation isn't permitted, you can still run `grab_window_bs4.py` manually for demonstration, or rely on official Open Data APIs and use the camera only as an embedded view in your final app.

---

## 1) Install
Python 3.11 recommended.

```bash
pip install requests beautifulsoup4
```

For later modeling:
```bash
pip install pandas scikit-learn ultralytics opencv-python
```

Prereqs: note the project expects Python 3.11 (point to winget install --id Python.Python.3.11 -e or python.org).

Create/Activate env: run from the outer folder:

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Project-1-M31-main\requirements.txt scikit-learn joblib

Project folder reminder: sources live in Project-1-M31-main; run commands there.
Data pipeline order:
python src\clean_data.py → produces TRAFFIC_CLEANED.csv
python src\train_linear_regression.py / train_knn.py / train_decision_tree.py → creates the model .pkls.
Optional YOLO workflow: python src\yolo_count.py, python src\build_dataset.py, etc., noting outputs.
Troubleshooting: reactivate the env in new shells; reinstall if pip errors mention Python 3.14.

---

## 2) Configure `config.json`
Open `config.json` and change:
- `camera_page_url`: the web page that **contains** the <img> snapshot (or similar).
- `image_css_selector`: a **CSS selector** that points to the `<img>` element. Example: `img#camera` or `img.snapshot`.
- `cadence_sec`: seconds between frames inside the 20-minute window (10–15 sec is fine).
- `window_minutes`: usually **20** per your professor.
- `run_times_local`: daily times to start windows (local time). Example: `["08:00","13:00","18:00"]`.
- `total_days`: how many days to run (e.g., 14 or 21).
- `user_agent`: polite custom UA string for HTTP requests.

**Tip:** Start with 1–2 manual test windows before enabling the scheduler.

---

## 3) Run a single 20-minute window (manual)
```bash
python src/grab_window_bs4.py
```
Images are saved under `data/raw/images/YYYY/MM/DD/HHMMSS.jpg`.
Two CSV logs are written to `data/logs/`:
- `images_log.csv` (one row per image)
- `windows_log.csv` (one row per collection window)

---

## 4) Run the simple scheduler (no cron)
Edit `config.json` (times + days), then:

```bash
python src/scheduler_simple.py
```
Leave the terminal open. It will:
- start a collection window at each `run_times_local` time,
- run for `total_days`,
- then exit.

---

## 5) Deduplicate frozen frames (optional)
Some cameras freeze and serve identical frames. After a day of collection:

```bash
python src/utils_hash.py --dedupe data/raw/images
```
This writes `data/logs/duplicates.csv` and a summary. By default it **keeps** the earliest copy and **removes** later duplicates (you can switch to "report-only" mode).

---

## 6) Next steps after collection
- Auto-label images with a pretrained detector (YOLO) to get `vehicle_count` per timestamp.
- Build a training table: `[timestamp, vehicle_count, hour, dayofweek, ...]`.
- Train a baseline regressor to predict the count **+20 minutes** ahead.
- Report MAE/RMSE and show error by hour/day.

---

## 7) Train the models
Once `src/clean_data.py` has produced `TRAFFIC_CLEANED.csv`, fit the tabular models:

```powershell
python src/train_linear_regression.py
python src/train_knn.py
python src/train_decision_tree.py
python src/train_random_forest.py
```

Each script saves a `.pkl` in the project root. `train_random_forest.py` now reports the holdout accuracy while training and writes `model_random_forest.pkl`.

## 8) Launch the desktop GUI
Use the lightweight Tkinter GUI to try different models and feature inputs without touching the code:

```powershell
python src/gui.py
```

The GUI loads whichever `.pkl` files are available, displays their quick evaluation metrics (accuracy for classifiers, MAE for the regression model), and lets you enter `hour`/`day` pairs to preview predictions.
