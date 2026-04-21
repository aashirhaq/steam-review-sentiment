# Steam Review Analytics

A four-stage pipeline that fetches Steam game reviews, scores them with a
pre-trained sentiment model, aggregates the results, and displays everything
in an interactive Streamlit dashboard.

---

## What it does

| Stage | Script | Output |
|---|---|---|
| 1 — Fetch reviews | `fetch_reviews.py` | `output/reviews.csv` + `output/reviews.json` |
| 2 — Sentiment analysis | `sentiment_analysis.py` | `output/reviews_with_sentiment.csv` |
| 3 — Aggregate | `analyze_sentiment.py` | `output/sentiment_report.json` |
| 4 — Dashboard | `app.py` | Streamlit app at `http://localhost:8501` |

**Games covered:** Dota 2 · Counter-Strike 2 · Rust · Stardew Valley · Baldur's Gate 3

**Model:** [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (downloaded automatically from Hugging Face on first run, ~260 MB, cached locally afterwards)

---

## Project structure

```
steam-review-analytics/
├── run_pipeline.py            ← orchestration script (start here)
├── fetch_reviews.py           ← Stage 1: Steam API review fetcher
├── sentiment_analysis.py      ← Stage 2: DistilBERT sentiment scorer
├── analyze_sentiment.py       ← Stage 3: aggregation & word frequency
├── app.py                     ← Stage 4: Streamlit dashboard
├── requirements.txt
├── README.md
└── output/                    ← created automatically
    ├── reviews.csv
    ├── reviews.json
    ├── reviews_with_sentiment.csv
    └── sentiment_report.json
```

---

## Requirements

- Python **3.10 or newer**
- Internet access for the first run (Steam API + Hugging Face model download)
- No API keys required — the Steam review endpoint is public

> **GPU (optional):** If you have an NVIDIA GPU with CUDA, Stage 2 runs
> significantly faster. The script detects CUDA automatically; no extra
> configuration is needed.

---

## Step-by-step setup

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd steam-review-analytics
```

Or simply copy all `.py` files into a single folder.

---

### 2. Create a virtual environment (recommended)

**Windows (Command Prompt / PowerShell):**
```bat
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> If you are using **Anaconda**, create a conda environment instead:
> ```bash
> conda create -n steam-analytics python=3.11
> conda activate steam-analytics
> ```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Used for |
|---|---|
| `requests` | Steam Store API calls |
| `steamreviews` | (imported but not used for pagination — kept for reference) |
| `transformers` | Hugging Face DistilBERT pipeline |
| `torch` | Neural-network inference backend |
| `streamlit` | Interactive dashboard |
| `pandas` | DataFrame construction for Streamlit charts |

> **Windows + Anaconda users:** if you see an `OMP: Error #15` crash during
> Stage 2, it is caused by a duplicate OpenMP runtime. The script sets
> `KMP_DUPLICATE_LIB_OK=TRUE` automatically — no action needed on your part.

---

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

The orchestrator runs all four stages in order and opens the dashboard when
done. Each stage is **automatically skipped** if its output file already
exists, so re-running the command is fast after the first run.

```
2026-04-20 10:00:01 [INFO] ===================================================
2026-04-20 10:00:01 [INFO]   Steam Review Analytics — Full Pipeline
2026-04-20 10:00:01 [INFO] ===================================================
2026-04-20 10:00:01 [INFO] ▶  Running Step 1: Fetch reviews …
2026-04-20 10:01:15 [INFO] ✔  Step 1: Fetch reviews complete.
2026-04-20 10:01:15 [INFO] ▶  Running Step 2: Sentiment analysis …
2026-04-20 10:07:00 [INFO] ✔  Step 2: Sentiment analysis complete.
2026-04-20 10:07:00 [INFO] ▶  Running Step 3: Aggregate sentiment report …
2026-04-20 10:07:01 [INFO] ✔  Step 3: Aggregate sentiment report complete.
2026-04-20 10:07:01 [INFO] ▶  Launching Streamlit dashboard …
2026-04-20 10:07:01 [INFO]     Open http://localhost:8501 in your browser.
```

Then open **http://localhost:8501** in your browser.

---

### 5. Use the dashboard

1. Use the **sidebar dropdown** to select a game.
2. The top row of **metric cards** shows total reviews, positive %, negative %, neutral %, and average model confidence.
3. The **Sentiment Breakdown** bar chart shows raw counts per label.
4. The **Confidence by Label** chart shows how certain the model was for positive vs negative reviews.
5. The **Top Praise Words** and **Top Complaint Words** sections show the most common words in positive and negative reviews respectively (stopwords and game-neutral words removed).
6. The **Cross-Game Comparison** section at the bottom lets you compare all five games side by side.

---

## Running individual stages

You can run any stage on its own:

```bash
# Stage 1 only — fetch raw reviews
python fetch_reviews.py

# Stage 2 only — score with DistilBERT (requires output/reviews.csv)
python sentiment_analysis.py

# Stage 3 only — compute aggregations (requires output/reviews_with_sentiment.csv)
python analyze_sentiment.py

# Stage 4 only — open dashboard (requires output/sentiment_report.json)
streamlit run app.py
```

---

## Pipeline flags

```
python run_pipeline.py --force       # re-run all stages, overwrite existing output
python run_pipeline.py --skip-app    # run stages 1–3 only, do not launch dashboard
python run_pipeline.py --force --skip-app  # re-run everything, no dashboard
```

---

## Customising the pipeline

### Fetch more (or fewer) reviews

Open `fetch_reviews.py` and change:

```python
MAX_REVIEWS_PER_GAME = 500   # set to 0 for unlimited (may take hours)
```

### Change the games

In `fetch_reviews.py`, edit the `GAMES` list:

```python
GAMES = [
    {"app_id": "570",     "name": "Dota 2"},
    {"app_id": "730",     "name": "Counter-Strike 2"},
    # add or replace entries here
    # find an app_id in the game's Steam store URL:
    # https://store.steampowered.com/app/1091500/  → app_id = "1091500" (Cyberpunk 2077)
]
```

### Change the sentiment model

In `sentiment_analysis.py`, change:

```python
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
```

Any Hugging Face `text-classification` model works here. Alternatives:

| Model | Notes |
|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 3-class (positive/neutral/negative), trained on tweets |
| `siebert/sentiment-roberta-large-english` | Larger, more accurate, slower |
| `finiteautomata/bertweet-base-sentiment-analysis` | BERTweet, good for short informal text |

### Adjust the neutral threshold

Reviews where the model confidence is below this value are labelled NEUTRAL
instead of POSITIVE or NEGATIVE. Edit in `sentiment_analysis.py`:

```python
NEUTRAL_THRESHOLD = 0.65   # lower = fewer neutrals, higher = more neutrals
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: output/reviews.csv` | Run Stage 1 first: `python fetch_reviews.py` |
| `OMP: Error #15` crash on Windows | Already handled automatically; if it persists, run `set KMP_DUPLICATE_LIB_OK=TRUE` before running the script |
| Hugging Face model download is slow | The model (~260 MB) is only downloaded once and cached in `~/.cache/huggingface/`. Subsequent runs load from disk. |
| `streamlit: command not found` | Run `pip install streamlit` or use `python -m streamlit run app.py` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` and open `http://localhost:8502` |
| Steam API returns no reviews | The Steam review endpoint is public and requires no key. If you get empty results, the app_id may be wrong or the game may have no English reviews. |

---

## No API keys needed

This project uses only:
- The **Steam Store review endpoint** (`store.steampowered.com/appreviews/`) — fully public, no authentication
- **Hugging Face Hub** for model weights — public download, no account required (an optional `HF_TOKEN` env var removes rate limits on the Hub but is not needed for this model)
