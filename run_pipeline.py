"""
Orchestration script — runs the full pipeline in order:

  Step 1  fetch_reviews.py      — pull Steam reviews via the Steam Store API
  Step 2  sentiment_analysis.py — score each review with DistilBERT
  Step 3  analyze_sentiment.py  — compute per-game aggregations
  Step 4  app.py                — launch the Streamlit dashboard

Each step is skipped automatically when its output already exists on disk,
so re-running the script after a partial failure picks up where it left off.
Pass --force to re-run every step from scratch.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — must match the constants inside each script
# ---------------------------------------------------------------------------

OUTPUT_DIR      = Path("output")
REVIEWS_CSV     = OUTPUT_DIR / "reviews.csv"
SENTIMENT_CSV   = OUTPUT_DIR / "reviews_with_sentiment.csv"
REPORT_JSON     = OUTPUT_DIR / "sentiment_report.json"

SCRIPTS = {
    "fetch":     Path("fetch_reviews.py"),
    "sentiment": Path("sentiment_analysis.py"),
    "analyze":   Path("analyze_sentiment.py"),
    "app":       Path("app.py"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_scripts() -> None:
    missing = [name for name, path in SCRIPTS.items() if not path.exists()]
    if missing:
        log.error("Missing script files: %s", ", ".join(missing))
        sys.exit(1)


def _run(script: Path, label: str) -> None:
    log.info("▶  Running %s …", label)
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
    )
    if result.returncode != 0:
        log.error("✖  %s failed (exit code %d). Aborting pipeline.", label, result.returncode)
        sys.exit(result.returncode)
    log.info("✔  %s complete.", label)


def _launch_app() -> None:
    log.info("▶  Launching Streamlit dashboard …")
    log.info("    Open http://localhost:8501 in your browser.")
    log.info("    Press Ctrl+C to stop.")
    # os.execv is unreliable on Windows/Git Bash; subprocess handles Ctrl+C correctly.
    # --server.headless true skips Streamlit's first-run email prompt.
    # --browser.gatherUsageStats false suppresses the usage-stats banner.
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", str(SCRIPTS["app"]),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        check=False,
    )

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_fetch(force: bool) -> None:
    if not force and REVIEWS_CSV.exists():
        log.info("⏭  Step 1 skipped — %s already exists.", REVIEWS_CSV)
        return
    _run(SCRIPTS["fetch"], "Step 1: Fetch reviews")


def step_sentiment(force: bool) -> None:
    if not force and SENTIMENT_CSV.exists():
        log.info("⏭  Step 2 skipped — %s already exists.", SENTIMENT_CSV)
        return
    if not REVIEWS_CSV.exists():
        log.error("Step 2 requires %s. Run Step 1 first.", REVIEWS_CSV)
        sys.exit(1)
    _run(SCRIPTS["sentiment"], "Step 2: Sentiment analysis")


def step_analyze(force: bool) -> None:
    if not force and REPORT_JSON.exists():
        log.info("⏭  Step 3 skipped — %s already exists.", REPORT_JSON)
        return
    if not SENTIMENT_CSV.exists():
        log.error("Step 3 requires %s. Run Step 2 first.", SENTIMENT_CSV)
        sys.exit(1)
    _run(SCRIPTS["analyze"], "Step 3: Aggregate sentiment report")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Steam Review Analytics pipeline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every step even if output files already exist.",
    )
    parser.add_argument(
        "--skip-app",
        action="store_true",
        help="Stop after Step 3 without launching the Streamlit app.",
    )
    args = parser.parse_args()

    _check_scripts()
    OUTPUT_DIR.mkdir(exist_ok=True)

    log.info("=" * 55)
    log.info("  Steam Review Analytics — Full Pipeline")
    log.info("=" * 55)

    step_fetch(args.force)
    step_sentiment(args.force)
    step_analyze(args.force)

    if args.skip_app:
        log.info("Pipeline complete. Dashboard not launched (--skip-app).")
        return

    _launch_app()   # does not return


if __name__ == "__main__":
    main()
