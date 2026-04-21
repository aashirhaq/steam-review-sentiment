# Line-by-Line Explanation: `sentiment_analysis.py`

---

## Lines 1–14 — Module Docstring

```python
"""
Runs sentiment analysis on Steam review texts using
distilbert-base-uncased-finetuned-sst-2-english and writes the results
to output/reviews_with_sentiment.csv.

Label mapping:
  POSITIVE  → confidence >= threshold
  NEGATIVE  → confidence >= threshold
  NEUTRAL   → model confidence < threshold  (low-signal / very short reviews)

Reviews longer than the model's 512-token limit are truncated to the first
512 tokens by the tokenizer (truncation=True), preserving as much signal
as possible without splitting into multiple chunks.
"""
```

A **docstring** is a string written at the very top of a file (before any code) that describes what the whole script does.
This one tells us three things:

- **What model is used** — `distilbert-base-uncased-finetuned-sst-2-english`, a pre-trained sentiment model from Hugging Face.
- **What the three labels mean** — POSITIVE and NEGATIVE come from the model when it is confident; NEUTRAL is assigned by *our own code* when the model's confidence is too low.
- **How long reviews are handled** — reviews with more than 512 tokens are automatically cut off (truncated) by the tokenizer rather than being split across multiple passes.

---

## Lines 16–20 — Standard Library Imports

```python
import csv
import logging
import os
import time
from pathlib import Path
```

These are all built into Python — no installation needed.

| Import | What it provides |
|---|---|
| `csv` | Read and write CSV files row by row |
| `logging` | Print timestamped progress messages instead of bare `print()` |
| `os` | Access and modify operating-system environment variables |
| `time` | Measure elapsed wall-clock time |
| `Path` (from `pathlib`) | Object-oriented file path manipulation that works on Windows, Mac, and Linux without string hacks |

---

## Lines 22–24 — OpenMP Duplicate Runtime Fix

```python
# Anaconda bundles its own OpenMP runtime; PyTorch ships another copy.
# This env var prevents a fatal crash on Windows when both are loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

**What is OpenMP?**
OpenMP is a library that lets programs use multiple CPU cores in parallel. Both Anaconda (the Python distribution) and PyTorch ship their own copy of it. When Python loads both, Windows detects the duplication and kills the process with exit code 3.

**What does `os.environ.setdefault` do?**
`os.environ` is a dictionary of all environment variables for the current process.
`setdefault("KEY", "VALUE")` means: *"Set this variable to VALUE, but only if it is not already set."* This is safer than a direct assignment because it does not override a value the user may have intentionally set before running the script.

**What does `KMP_DUPLICATE_LIB_OK=TRUE` tell Intel's runtime?**
It tells Intel's OpenMP library to continue even when a duplicate is detected, instead of aborting. The comment is important — this is an unsupported workaround, but it is the standard practice on Anaconda + PyTorch on Windows.

**Why is this line placed *before* `import torch`?**
Because the crash happens the moment `torch` is imported and loads the runtime. The environment variable must exist before that import runs.

---

## Lines 26–27 — Third-Party Imports

```python
import torch
from transformers import pipeline
```

**`torch`** is PyTorch — the deep-learning framework that actually runs the model calculations on your CPU or GPU.

**`pipeline`** from the `transformers` library (by Hugging Face) is a high-level wrapper that hides all the complexity of:
- downloading the model weights
- loading the tokenizer
- converting text to numbers (tokens)
- running the neural network forward pass
- converting the output numbers back to a human-readable label and score

You call `pipeline(...)` once to build it, then call the resulting object like a function to score text.

---

## Lines 29–33 — Logging Setup

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
```

**`logging.basicConfig(...)`** configures the global logging system once for the whole script.

- `level=logging.INFO` — show messages at INFO level and above (INFO, WARNING, ERROR). DEBUG messages are hidden.
- `format="%(asctime)s [%(levelname)s] %(message)s"` — each log line looks like:
  `2026-04-17 16:49:19,644 [INFO] Loaded 2500 reviews from output\reviews.csv`

**`logging.getLogger(__name__)`** creates a logger named after this module (which is `"__main__"` when run directly). Using a named logger instead of `print()` means you can later add file handlers, filter by module, or silence specific loggers without touching the code.

---

## Lines 39–47 — Configuration Constants

```python
INPUT_CSV  = Path("output/reviews.csv")
OUTPUT_CSV = Path("output/reviews_with_sentiment.csv")

MODEL_NAME          = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE          = 64
NEUTRAL_THRESHOLD   = 0.65
EMPTY_LABEL         = "NEUTRAL"
EMPTY_CONFIDENCE    = 0.0
LOG_EVERY           = 250
```

All tunable values live here at the top so they are easy to find and change.

| Constant | Meaning |
|---|---|
| `INPUT_CSV` | Path to the CSV written by `fetch_reviews.py` |
| `OUTPUT_CSV` | Path where the enriched CSV will be written |
| `MODEL_NAME` | The exact Hugging Face model identifier to download/load |
| `BATCH_SIZE` | How many reviews to send to the model in one forward pass. Larger = faster if you have VRAM; 64 is safe on CPU |
| `NEUTRAL_THRESHOLD` | If the model's confidence is below 0.65, we override the label to NEUTRAL regardless of what the model said |
| `EMPTY_LABEL` | The label assigned to blank or whitespace-only review texts |
| `EMPTY_CONFIDENCE` | The confidence score assigned to blank reviews |
| `LOG_EVERY` | Print a progress line every 250 reviews processed |

**Why `Path(...)` and not a plain string?**
`Path` objects have methods like `.exists()`, `.open()`, `.parent`, and `.mkdir()` that you can chain cleanly. They also handle Windows backslashes vs Unix forward slashes automatically.

---

## Lines 53–56 — `_label()` Helper Function

```python
def _label(raw_label: str, confidence: float) -> str:
    if confidence < NEUTRAL_THRESHOLD:
        return "NEUTRAL"
    return raw_label.upper()
```

**What this function does:**
The DistilBERT SST-2 model only knows two classes: `"POSITIVE"` and `"NEGATIVE"`. It always picks one — even when the review is too short to judge, written in another language, or is a meaningless string like `"yyu"`.

This function adds a third outcome. If the model's winning score is below 0.65, it means the model is not confident — so we return `"NEUTRAL"` instead of a potentially wrong label.

**`raw_label.upper()`** — the model sometimes returns `"POSITIVE"` and sometimes `"LABEL_1"` depending on the model version. Calling `.upper()` normalises capitalisation, so the output is always consistent.

**The leading underscore `_label`** is a Python convention meaning "this is a private helper, not part of the public interface of this module."

---

## Lines 59–69 — `_load_pipeline()` Function

```python
def _load_pipeline() -> pipeline:
    device = 0 if torch.cuda.is_available() else -1
    device_name = f"GPU (cuda:{device})" if device >= 0 else "CPU"
    log.info("Loading model '%s' on %s", MODEL_NAME, device_name)
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=device,
        truncation=True,
        max_length=512,
    )
```

**Line 60 — GPU detection:**
`torch.cuda.is_available()` returns `True` if a CUDA-capable NVIDIA GPU is found. If yes, `device = 0` (the first GPU). If no, `device = -1`, which tells the pipeline to use the CPU.

**Line 61 — Human-readable device name:**
This is only used in the log message so you can see at a glance whether the model loaded onto the GPU or CPU.

**Line 63–69 — `pipeline(...)` call:**

| Argument | Effect |
|---|---|
| `"sentiment-analysis"` | Tells Hugging Face what *type* of task this is |
| `model=MODEL_NAME` | Downloads and loads this specific model from the Hub (cached locally after first run) |
| `device=device` | Puts the model on GPU or CPU |
| `truncation=True` | If input text is longer than `max_length` tokens, silently cut it down instead of raising an error |
| `max_length=512` | DistilBERT's hard limit is 512 tokens. Setting this explicitly prevents a warning |

---

## Lines 72–75 — `_run_batch()` Function

```python
def _run_batch(nlp, texts: list[str]) -> list[tuple[str, float]]:
    """Returns list of (label, confidence) for each text in the batch."""
    results = nlp(texts, batch_size=BATCH_SIZE, truncation=True)
    return [(_label(r["label"], r["score"]), round(r["score"], 4)) for r in results]
```

**`nlp(texts, ...)`** — calling the pipeline object like a function runs inference on every string in the list. It returns a list of dicts like:
```python
[{"label": "POSITIVE", "score": 0.9987}, {"label": "NEGATIVE", "score": 0.7234}, ...]
```

**`batch_size=BATCH_SIZE`** — even though we are already slicing into batches outside, passing `batch_size` here lets the pipeline's internal DataLoader manage GPU memory efficiently.

**List comprehension on line 75** — for every result dict `r` in the list:
- `_label(r["label"], r["score"])` — apply the NEUTRAL threshold logic
- `round(r["score"], 4)` — keep 4 decimal places (e.g. `0.9987`)
- Returns a list of `(label, confidence)` tuples, one per input text

---

## Lines 81–83 — `main()` — Input Validation

```python
def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}. Run fetch_reviews.py first.")
```

`INPUT_CSV.exists()` checks whether the file is actually on disk before doing anything. If it is missing, a `FileNotFoundError` is raised with a helpful message telling the user what to do. This is much cleaner than letting Python crash with a confusing `[Errno 2]` error three steps later.

---

## Lines 85–87 — Load the CSV into Memory

```python
    with INPUT_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info("Loaded %d reviews from %s", len(rows), INPUT_CSV)
```

**`INPUT_CSV.open(encoding="utf-8")`** — opens the file with UTF-8 encoding, which is required because Steam reviews contain characters from many languages (Cyrillic, CJK, emoji, etc.).

**`csv.DictReader(f)`** — reads each CSV row as a Python `dict` where the keys are the column names from the header row. So each row looks like:
```python
{"game_name": "Dota 2", "review_text": "Playing since 2003", "recommended": "True", ...}
```

**`list(...)`** — `DictReader` is a lazy iterator; wrapping it in `list()` reads the entire file into RAM immediately and closes the file handle when the `with` block exits.

---

## Line 89 — Load the Model

```python
    nlp = _load_pipeline()
```

This is the most expensive line in the script. It:
1. Checks the local Hugging Face cache (`~/.cache/huggingface/hub/`)
2. Downloads the model weights (~260 MB) from the Hub on first run
3. Loads the weights into memory (CPU RAM or GPU VRAM)
4. Initialises the tokenizer

After this line, `nlp` is a callable object that accepts text and returns sentiment predictions.

---

## Line 91 — Build Output Column List

```python
    output_fields = list(rows[0].keys()) + ["sentiment_label", "sentiment_confidence"]
```

`rows[0].keys()` returns the column names from the first row — which are the original CSV columns. We append two new column names: `"sentiment_label"` and `"sentiment_confidence"`. This list is later used to define the header row of the output CSV.

---

## Lines 93–95 — Pre-allocate Result Lists

```python
    texts       = [r["review_text"] for r in rows]
    sentiments  = [""] * len(rows)
    confidences = [0.0] * len(rows)
```

**`texts`** — a flat list of every review text string, in the same order as `rows`. This is what the model receives.

**`sentiments`** and **`confidences`** — pre-allocated lists of the same length, filled with empty defaults. We will fill in the real values by index as batches complete. Pre-allocating by index (rather than `.append()`) keeps results aligned with the original row order even when batches are processed.

---

## Lines 97–99 — Timer and Batch Range

```python
    t0      = time.time()
    total   = len(texts)
    batches = range(0, total, BATCH_SIZE)
```

**`time.time()`** — records the start time as a Unix timestamp (seconds since 1970-01-01). Used later to compute elapsed time.

**`range(0, total, BATCH_SIZE)`** — generates the starting index of each batch: `0, 64, 128, 192, ...`. For 2500 reviews with `BATCH_SIZE=64` this produces 40 values (the last batch will be smaller than 64 if 2500 is not divisible by 64).

---

## Lines 101–102 — Slice the Batch

```python
    for batch_start in batches:
        batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
```

On each loop iteration, `batch_start` is the starting index (0, 64, 128, ...). `batch_texts` is a slice of up to 64 texts from the full list. Python's slice syntax handles the last batch automatically — if fewer than 64 texts remain, it just returns however many are left.

---

## Lines 105–107 — Separate Empty Reviews

```python
        indices        = list(range(batch_start, batch_start + len(batch_texts)))
        non_empty_idx  = [i for i, t in zip(indices, batch_texts) if t.strip()]
        non_empty_text = [texts[i] for i in non_empty_idx]
```

**Why this matters:** The Hugging Face pipeline raises an error if you pass it an empty string `""`. Some Steam reviews have no text (the user just clicked thumbs up/down). We need to filter them out before calling the model.

**`indices`** — the absolute row indices for this batch (e.g. `[0, 1, 2, ..., 63]` for the first batch).

**`non_empty_idx`** — keeps only those indices where `t.strip()` is truthy, i.e. the text is not blank or whitespace-only. `zip(indices, batch_texts)` pairs each global index with its text string.

**`non_empty_text`** — the actual text strings for non-empty reviews, used as input to the model.

---

## Lines 109–113 — Run Inference on Non-Empty Reviews

```python
        if non_empty_text:
            results = _run_batch(nlp, non_empty_text)
            for i, (label, conf) in zip(non_empty_idx, results):
                sentiments[i]  = label
                confidences[i] = conf
```

**`if non_empty_text:`** — skip the model call entirely if every review in this batch is blank (edge case).

**`_run_batch(nlp, non_empty_text)`** — sends the texts to the model and gets back a list of `(label, confidence)` tuples.

**`zip(non_empty_idx, results)`** — pairs each result back with its original global row index, so we write the sentiment into the right position in `sentiments` and `confidences`.

---

## Lines 116–119 — Fill in Empty Review Defaults

```python
        for i in indices:
            if not texts[i].strip():
                sentiments[i]  = EMPTY_LABEL
                confidences[i] = EMPTY_CONFIDENCE
```

For every index in the current batch where the text *is* blank, write `"NEUTRAL"` and `0.0`. This loop runs after the model call, so it only touches positions that were skipped above.

---

## Lines 121–125 — Progress Logging

```python
        done = min(batch_start + BATCH_SIZE, total)
        if done % LOG_EVERY < BATCH_SIZE or done == total:
            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed else 0
            log.info("Progress: %d / %d  (%.1f reviews/s)", done, total, rate)
```

**`done`** — how many reviews have been processed so far. `min(...)` handles the last partial batch correctly.

**`done % LOG_EVERY < BATCH_SIZE`** — this condition fires the first time `done` crosses a multiple of `LOG_EVERY` (250). For example: when `done` goes from 192 to 256, `256 % 250 = 6`, which is less than `BATCH_SIZE=64`, so the log fires. This avoids logging every single batch while still giving regular updates.

**`or done == total`** — always log the final batch, even if it doesn't land exactly on a LOG_EVERY boundary.

**`rate`** — reviews processed per second. The `if elapsed else 0` guard prevents a ZeroDivisionError on the extremely unlikely event the first batch completes in 0 seconds.

---

## Lines 127–132 — Write Output CSV

```python
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row, label, conf in zip(rows, sentiments, confidences):
            writer.writerow({**row, "sentiment_label": label, "sentiment_confidence": conf})
```

**`OUTPUT_CSV.parent.mkdir(exist_ok=True)`** — creates the `output/` directory if it does not exist. `exist_ok=True` means do not raise an error if it already exists.

**`open("w", newline="", encoding="utf-8")`** — opens for writing. `newline=""` is required by Python's `csv` module on Windows to prevent it writing an extra blank line between every row.

**`csv.DictWriter(f, fieldnames=output_fields)`** — a writer that takes dicts and maps keys to CSV columns using the `fieldnames` list as the column order.

**`writer.writeheader()`** — writes the column name row at the top of the file.

**`{**row, "sentiment_label": label, "sentiment_confidence": conf}`** — the `**row` syntax "unpacks" all the original key-value pairs from the input row, then adds (or overwrites) the two new keys. This is equivalent to copying the dict and adding two new entries.

---

## Lines 134–139 — Final Summary Log

```python
    elapsed = time.time() - t0
    pos  = sentiments.count("POSITIVE")
    neg  = sentiments.count("NEGATIVE")
    neu  = sentiments.count("NEUTRAL")
    log.info("Done in %.1fs — POSITIVE: %d  NEGATIVE: %d  NEUTRAL: %d", elapsed, pos, neg, neu)
    log.info("Output saved → %s", OUTPUT_CSV)
```

**`time.time() - t0`** — total wall-clock seconds since the timer was started.

**`sentiments.count("POSITIVE")`** — Python list `.count()` scans the list and returns how many times the value appears.

The final two log lines summarise the run: total time, label distribution, and the output file path.

---

## Lines 142–143 — Entry Point Guard

```python
if __name__ == "__main__":
    main()
```

**What this means:** `__name__` is a special Python variable. When you run a file directly with `python sentiment_analysis.py`, Python sets `__name__` to `"__main__"`. If instead this file were *imported* by another script (e.g. `import sentiment_analysis`), `__name__` would be `"sentiment_analysis"` and `main()` would *not* be called automatically.

This pattern makes the file safe to import as a library without triggering a full run — and it is considered best practice in every Python script that has a `main()` function.

---

## Data Flow Summary

```
reviews.csv
    │
    ▼
csv.DictReader  →  rows (list of dicts)
                        │
                        ├── texts (list of strings)
                        │       │
                        │       ▼
                        │   [filter empties]
                        │       │
                        │       ▼
                        │   pipeline (DistilBERT)
                        │       │
                        │       ▼
                        │   _label() → NEUTRAL threshold check
                        │       │
                        ▼       ▼
                   sentiments[], confidences[]
                        │
                        ▼
                csv.DictWriter  →  reviews_with_sentiment.csv
```
