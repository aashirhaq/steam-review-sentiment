# Steam Review Sentiment

A pipeline that fetches Steam game reviews, scores them with a pre-trained sentiment model, aggregates the results, and displays everything in an interactive Streamlit dashboard.

**Live app:** [steam-review-sentiment-xt4bsraqrvdp8gwakq4sj6.streamlit.app](https://steam-review-sentiment-xt4bsraqrvdp8gwakq4sj6.streamlit.app/)

---

## How it works

### Local pipeline (3 stages)

| Stage | Script | Output |
|---|---|---|
| 1 — Fetch reviews | `fetch_reviews.py` | `output/reviews.csv` + `output/reviews.json` |
| 2 — Sentiment analysis | `sentiment_analysis.py` | `output/reviews_with_sentiment.csv` |
| 3 — Aggregate | `analyze_sentiment.py` | `output/sentiment_report.json` |

### Cloud deployment

```
GitHub Actions (daily at 3am UTC)
  → runs all 3 pipeline stages
  → uploads sentiment_report.json to GitHub Release "data-latest"

Streamlit Cloud (on app start)
  → downloads sentiment_report.json from the release
  → renders the dashboard
```

Data is refreshed automatically every day. No ML dependencies run on Streamlit Cloud — it only downloads a small JSON file.

**Games covered:** Dota 2 · Counter-Strike 2 · Rust · Stardew Valley · Baldur's Gate 3

**Model:** [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (~260 MB, downloaded automatically from Hugging Face on first run and cached locally)

---

## Project structure

```
steam-review-sentiment/
├── .github/
│   └── workflows/
│       └── refresh-data.yml       ← daily GitHub Actions pipeline
├── fetch_reviews.py               ← Stage 1: Steam API review fetcher
├── sentiment_analysis.py          ← Stage 2: DistilBERT sentiment scorer
├── analyze_sentiment.py           ← Stage 3: aggregation & word frequency
├── run_pipeline.py                ← orchestration script (run locally)
├── app.py                         ← Streamlit dashboard
├── requirements.txt               ← app deps only (streamlit, pandas)
├── requirements-pipeline.txt      ← pipeline deps (torch, transformers, etc.)
└── output/                        ← created automatically, gitignored
    ├── reviews.csv
    ├── reviews.json
    ├── reviews_with_sentiment.csv
    └── sentiment_report.json
```

---

## Local setup

### 1. Clone the repo

```bash
git clone https://github.com/aashirhaq/steam-review-sentiment
cd steam-review-sentiment
```

### 2. Create a virtual environment

**Windows:**
```bat
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install pipeline dependencies

```bash
pip install -r requirements-pipeline.txt
```

| Package | Used for |
|---|---|
| `requests` | Steam Store API calls |
| `transformers` | Hugging Face DistilBERT pipeline |
| `torch` | Neural-network inference backend |
| `pandas` | Data processing |

> **GPU (optional):** Stage 2 detects CUDA automatically and runs on GPU if available — no extra configuration needed.

> **Windows + Anaconda:** If you see an `OMP: Error #15` crash during Stage 2, the script handles it automatically via `KMP_DUPLICATE_LIB_OK=TRUE`.

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

Each stage is automatically skipped if its output already exists, so re-running is fast after the first run.

```
[INFO] ▶  Running Step 1: Fetch reviews …
[INFO] ✔  Step 1 complete.
[INFO] ▶  Running Step 2: Sentiment analysis …
[INFO] ✔  Step 2 complete.
[INFO] ▶  Running Step 3: Aggregate sentiment report …
[INFO] ✔  Step 3 complete.
```

### 5. Launch the dashboard locally

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Running individual stages

```bash
python fetch_reviews.py        # Stage 1 — fetch raw reviews
python sentiment_analysis.py   # Stage 2 — score with DistilBERT (requires reviews.csv)
python analyze_sentiment.py    # Stage 3 — aggregate (requires reviews_with_sentiment.csv)
```

**Pipeline flags:**

```bash
python run_pipeline.py --force         # re-run all stages, overwrite existing output
python run_pipeline.py --skip-app      # run stages 1–3 only, do not launch dashboard
```

---

## Cloud deployment

### Full flow

```
┌─────────────────────────────────────────────────────┐
│              GitHub Actions  (daily 3am UTC)         │
│                                                       │
│  ubuntu runner spins up                               │
│    → pip install requirements-pipeline.txt            │
│    → python fetch_reviews.py   (hits Steam API)       │
│    → python sentiment_analysis.py  (runs DistilBERT)  │
│    → python analyze_sentiment.py   (builds JSON)      │
│    → gh release create data-latest sentiment_report.json │
└──────────────────────┬──────────────────────────────┘
                       │ uploads asset
                       ▼
         GitHub Release  "data-latest"
           └── sentiment_report.json
                       │
                       │ downloaded on cold start
                       ▼
┌──────────────────────────────────────────────────────┐
│              Streamlit Cloud                          │
│                                                       │
│  app.py starts                                        │
│    → output/sentiment_report.json missing?            │
│    → urllib.request downloads it from the release     │
│    → st.rerun() → dashboard renders                   │
└──────────────────────────────────────────────────────┘
```

### GitHub Actions — how it works

GitHub Actions is a CI/CD service built into GitHub. You define workflows as YAML files inside `.github/workflows/`. GitHub reads these files automatically — no external setup needed.

The workflow file for this project is `.github/workflows/refresh-data.yml`. It is triggered two ways:

```yaml
on:
  schedule:
    - cron: '0 3 * * *'   # automatically every day at 3am UTC
  workflow_dispatch:        # manually from the GitHub Actions tab
```

When it runs, GitHub spins up a free Ubuntu virtual machine and executes the steps in order:

| Step | What it does |
|---|---|
| `actions/checkout@v4` | Clones your repo onto the runner |
| `actions/setup-python@v5` | Installs Python 3.11 |
| `actions/cache@v4` | Caches pip packages to speed up future runs |
| `pip install -r requirements-pipeline.txt` | Installs torch, transformers, requests, pandas |
| `python fetch_reviews.py` | Fetches 500 recent reviews per game from the Steam API |
| `python sentiment_analysis.py` | Scores every review with DistilBERT on CPU |
| `python analyze_sentiment.py` | Aggregates results into `sentiment_report.json` |
| `gh release delete data-latest` | Removes the previous release (ignores error if none exists) |
| `gh release create data-latest ...` | Creates a new release and attaches `sentiment_report.json` |

The `gh` CLI is pre-installed on all GitHub-hosted runners. The workflow uses the built-in `GITHUB_TOKEN` — no secrets or extra configuration required.

**Typical runtime:** ~20–30 minutes per run (mostly Stage 2 — DistilBERT on CPU scoring 2,500 reviews).

**Cost:** Free. Public repos have unlimited GitHub Actions minutes. Private repos get 2,000 free minutes/month (this pipeline uses ~25 min/day = ~750 min/month).

**First run:** The daily schedule won't fire until the next 3am UTC. To create the initial release immediately:
> GitHub repo → **Actions** → **Refresh Steam Sentiment Data** → **Run workflow** → **Run workflow**

### Streamlit Cloud — how it works

Streamlit Cloud hosts the `app.py` dashboard. It installs only `requirements.txt` (`streamlit` + `pandas`) — the heavy ML packages are never installed there.

When the app starts:
1. `app.py` checks if `output/sentiment_report.json` exists locally
2. If not (always true on a fresh container), it downloads it from the GitHub Release using Python's built-in `urllib.request` — no extra dependencies
3. `st.rerun()` reloads the page, which now finds the file and renders the full dashboard

The download URL is hardcoded in `app.py`:
```python
DATA_URL = "https://github.com/aashirhaq/steam-review-sentiment/releases/download/data-latest/sentiment_report.json"
```

Streamlit Cloud redeploys automatically on every push to `main`. No manual redeployment is ever needed.

To deploy your own fork:
1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Select your repo, branch `main`, main file `app.py`
3. Click **Deploy**

---

## Customising the pipeline

### Fetch more (or fewer) reviews

In `fetch_reviews.py`:

```python
MAX_REVIEWS_PER_GAME = 500   # set to 0 for unlimited (may take hours)
```

### Change the games

In `fetch_reviews.py`, edit the `GAMES` list:

```python
GAMES = [
    {"app_id": "570",     "name": "Dota 2"},
    {"app_id": "730",     "name": "Counter-Strike 2"},
    # find app_id in the Steam store URL:
    # https://store.steampowered.com/app/1091500/ → app_id = "1091500"
]
```

### Change the sentiment model

In `sentiment_analysis.py`:

```python
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
```

Any Hugging Face `text-classification` model works. Alternatives:

| Model | Notes |
|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 3-class, trained on tweets |
| `siebert/sentiment-roberta-large-english` | More accurate, slower |
| `finiteautomata/bertweet-base-sentiment-analysis` | Good for short informal text |

### Adjust the neutral threshold

```python
NEUTRAL_THRESHOLD = 0.65   # lower = fewer neutrals, higher = more neutrals
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: output/reviews.csv` | Run Stage 1 first: `python fetch_reviews.py` |
| `OMP: Error #15` on Windows | Handled automatically. If it persists, set `KMP_DUPLICATE_LIB_OK=TRUE` before running. |
| Slow Hugging Face model download | The model (~260 MB) downloads once and caches in `~/.cache/huggingface/` |
| `streamlit: command not found` | Run `python -m streamlit run app.py` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| App shows "Failed to download data" | Trigger the GitHub Actions workflow manually to create the initial release |

---

## No API keys needed

- **Steam Store review endpoint** (`store.steampowered.com/appreviews/`) — fully public, no authentication
- **Hugging Face Hub** — public model download, no account required
