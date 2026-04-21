"""
Fetches Steam reviews for five hardcoded games via the Steam Store API and saves
them to output/reviews.csv and output/reviews.json.

Pagination uses Steam's cursor-based system; MAX_REVIEWS_PER_GAME caps how many
reviews are collected per game so the script finishes in a reasonable time.
Set MAX_REVIEWS_PER_GAME = 0 to fetch every available review (may take hours for
large titles like Dota 2 or CS2).
"""

import csv
import json
import time
import datetime
import logging
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMES = [
    {"app_id": "570",     "name": "Dota 2"},
    {"app_id": "730",     "name": "Counter-Strike 2"},
    {"app_id": "252490",  "name": "Rust"},
    {"app_id": "413150",  "name": "Stardew Valley"},
    {"app_id": "1086940", "name": "Baldur's Gate 3"},
]

MAX_REVIEWS_PER_GAME = 500   # 0 = unlimited (fetch all)
BATCH_SIZE           = 100   # Steam API max per request
BETWEEN_BATCH_PAUSE  = 1.0   # seconds between page requests
BETWEEN_GAME_PAUSE   = 3.0   # seconds between games
MAX_RETRIES          = 3
RETRY_BACKOFF        = 1.5   # urllib3 exponential backoff factor

STEAM_REVIEW_URL = "https://store.steampowered.com/appreviews/{app_id}"

BASE_PARAMS = {
    "json":           "1",
    "language":       "english",
    "review_type":    "all",
    "purchase_type":  "all",
    "filter":         "recent",
    "num_per_page":   str(BATCH_SIZE),
}

OUTPUT_DIR = Path("output")
CSV_PATH   = OUTPUT_DIR / "reviews.csv"
JSON_PATH  = OUTPUT_DIR / "reviews.json"

CSV_FIELDS = [
    "game_name",
    "app_id",
    "review_id",
    "review_text",
    "review_date",
    "recommended",
    "playtime_at_review_hours",
    "playtime_forever_hours",
    "votes_helpful",
    "votes_funny",
    "author_steamid",
]

# ---------------------------------------------------------------------------
# HTTP session with automatic retries on transient errors
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "steam-review-fetcher/1.0"})
    return session

SESSION = _build_session()

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_review(raw: dict, game_name: str, app_id: str) -> dict:
    author    = raw.get("author", {})
    timestamp = raw.get("timestamp_created", 0)
    review_date = (
        datetime.datetime.fromtimestamp(timestamp, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
        if timestamp else ""
    )
    return {
        "game_name":                game_name,
        "app_id":                   app_id,
        "review_id":                raw.get("recommendationid", ""),
        "review_text":              raw.get("review", "").replace("\n", " ").strip(),
        "review_date":              review_date,
        "recommended":              raw.get("voted_up"),
        "playtime_at_review_hours": round(author.get("playtime_at_review", 0) / 60, 2),
        "playtime_forever_hours":   round(author.get("playtime_forever",   0) / 60, 2),
        "votes_helpful":            raw.get("votes_helpful", 0),
        "votes_funny":              raw.get("votes_funny",   0),
        "author_steamid":           author.get("steamid", ""),
    }

# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _fetch_page(app_id: str, cursor: str) -> tuple[list[dict], str, int]:
    """
    Returns (reviews_list, next_cursor, total_reviews).
    Raises requests.HTTPError on non-2xx responses that exhaust retries.
    """
    params = {**BASE_PARAMS, "cursor": cursor}
    url    = STEAM_REVIEW_URL.format(app_id=app_id)

    resp = SESSION.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    if data.get("success") != 1:
        raise ValueError(f"Steam API returned success={data.get('success')} for app {app_id}")

    reviews      = data.get("reviews", [])
    next_cursor  = data.get("cursor", "")
    total        = data.get("query_summary", {}).get("total_reviews", 0)
    return reviews, next_cursor, total


def fetch_game_reviews(app_id: str, game_name: str) -> list[dict]:
    collected: list[dict] = []
    cursor    = "*"
    total     = None
    page      = 0

    while True:
        page += 1
        try:
            raw_reviews, next_cursor, total_reviews = _fetch_page(app_id, cursor)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 429:
                log.warning("Rate-limited on %s page %d — waiting 60s", game_name, page)
                time.sleep(60)
                continue
            log.error("HTTP %s fetching %s page %d — stopping early: %s", status, game_name, page, exc)
            break
        except (requests.exceptions.RequestException, ValueError) as exc:
            log.error("Error fetching %s page %d — stopping early: %s", game_name, page, exc)
            break

        if total is None:
            total = total_reviews
            log.info("%s — total reviews available: %d", game_name, total)

        collected.extend(_parse_review(r, game_name, app_id) for r in raw_reviews)

        fetched = len(collected)
        log.info("%s — fetched %d / %d", game_name, fetched, total)

        at_limit   = MAX_REVIEWS_PER_GAME and fetched >= MAX_REVIEWS_PER_GAME
        no_more    = not raw_reviews or not next_cursor or next_cursor == cursor
        if at_limit or no_more:
            break

        cursor = next_cursor
        time.sleep(BETWEEN_BATCH_PAUSE)

    if MAX_REVIEWS_PER_GAME:
        collected = collected[:MAX_REVIEWS_PER_GAME]

    log.info("%s — done, collected %d reviews", game_name, len(collected))
    return collected

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV saved → %s  (%d rows)", CSV_PATH, len(rows))


def save_json(rows: list[dict]) -> None:
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    log.info("JSON saved → %s  (%d records)", JSON_PATH, len(rows))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_rows: list[dict] = []

    for i, game in enumerate(GAMES):
        rows = fetch_game_reviews(game["app_id"], game["name"])
        all_rows.extend(rows)

        if i < len(GAMES) - 1:
            log.info("Pausing %.1fs before next game…", BETWEEN_GAME_PAUSE)
            time.sleep(BETWEEN_GAME_PAUSE)

    if not all_rows:
        log.error("No reviews collected — check network or app IDs.")
        return

    save_csv(all_rows)
    save_json(all_rows)
    log.info("Done. Total reviews collected: %d", len(all_rows))


if __name__ == "__main__":
    main()
