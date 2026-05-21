# Conversational Traffic Assistant — Aragó Street, Barcelona

A bilingual (English / Spanish) chatbot that forecasts traffic conditions on Aragó Street in Barcelona. Users ask natural-language questions; the system parses the query, fetches live weather, runs a Random Forest classifier, and replies in plain language via Google Gemini.

**Group 3 — Universidad Europea de Madrid · Final Delivery 2025/2026**
Iancu David · Roua Alarabe · Hassineen Al-abboodi

---

## Quick start

Requires Python 3.11.

```bash
# 1. Create and activate a virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate      # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set the Gemini API key (optional — without it the app uses template fallback)
$env:GEMINI_API_KEY="your_key_here"     # Windows PowerShell
# export GEMINI_API_KEY="your_key_here" # macOS / Linux

# 4. Run the app
streamlit run app.py
```

The app opens in your browser at <http://localhost:8501>.

---

## What's in the repo

| File | Role |
|---|---|
| `app.py` | Streamlit chatbot UI — the entry point |
| `src/nlp_parser.py` | Regex-based bilingual parser (extracts hour, day, intent) |
| `src/connector.py` | Bridges the parser to the Random Forest; fetches live weather from Open-Meteo |
| `src/llm_enhancer.py` | Wraps Google Gemini Flash Lite; bilingual response generation + scope filter |
| `model_random_forest.pkl` | Trained classifier (300 trees, 11 features, ~72% holdout accuracy) |
| `TRAFFIC_ENHANCED.csv` | Dataset of 2,161 records used to compute lag-feature medians at startup |
| `requirements.txt` | Python dependencies |

---

## How it works

```
User query
   → Scope filter (on-topic check)
   → NLP parser (hour · day · intent)
   → Model Connector (live weather + dataset-median lags → 11 features)
   → Random Forest (Low / Medium / High + confidence)
   → Gemini Flash Lite (2–3 sentence reply in the user's language)
   → Streamlit UI (chat bubble + traffic badge + weather caption)
```

Every layer has a fallback: if Open-Meteo fails, weather defaults to dataset medians; if Gemini fails, a template response is used; if the query is off-topic, the scope filter rejects it before any model or API call.

---

## Prior semester

The dataset and Random Forest classifier were produced in the previous semester by the same group. That work — image scraping from a Barcelona traffic camera, YOLOv8 vehicle counting, weather enrichment, model training — is not part of this repository. Only the runtime artifacts needed by the chatbot (`model_random_forest.pkl` and `TRAFFIC_ENHANCED.csv`) are kept here.

---

## Deployment

The app runs on the LORCA university VM (Linux, Python 3.11). Access is via SSH tunnel forwarding `localhost:8501` to the VM's port 8501 — see the final delivery document for details.
