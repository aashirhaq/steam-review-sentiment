"""
Reads reviews_with_sentiment.csv and produces a structured JSON report with:
  - total review count per game
  - positive / negative / neutral percentage per game
  - average sentiment confidence score per game
  - top-10 most common words in negative reviews per game
  - top-10 most common words in positive reviews per game

Output: output/sentiment_report.json
"""

import csv
import json
import logging
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV   = Path("output/reviews_with_sentiment.csv")
OUTPUT_JSON = Path("output/sentiment_report.json")

TOP_N_WORDS = 10

# Words to exclude from word-frequency analysis — common English filler that
# carries no sentiment signal.
STOPWORDS = {
    "the","a","an","and","or","but","is","it","in","on","at","to","for",
    "of","with","this","that","was","are","i","you","he","she","they","we",
    "my","your","its","not","be","been","have","has","had","do","did","does",
    "so","as","if","by","from","up","out","can","will","just","about","more",
    "very","game","games","steam","review","also","get","got","would","could",
    "should","s","t","m","re","ve","ll","d","no","all","there","their","when",
    "what","like","one","time","then","than","even","into","only","some","now",
    "other","after","which","how","who","much","any","still","over","back",
    "really","good","great","bad","play","played","playing","played","hours",
    "people","every","well","way",
}

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")   # match anything that is not word-char or space

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, drop stopwords and short tokens."""
    text  = _PUNCT_RE.sub(" ", text.lower())
    words = text.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def top_words(word_list: list[str], n: int = TOP_N_WORDS) -> list[dict]:
    """Return the top-N words as [{"word": ..., "count": ...}, ...]."""
    return [
        {"word": word, "count": count}
        for word, count in Counter(word_list).most_common(n)
    ]

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_report(rows: list[dict]) -> dict:
    # Bucket rows by game
    by_game: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_game[row["game_name"]].append(row)

    games_stats = {}

    for game_name, game_rows in sorted(by_game.items()):
        total = len(game_rows)

        # Label counts
        label_counts: Counter = Counter(r["sentiment_label"] for r in game_rows)
        pos_count = label_counts.get("POSITIVE", 0)
        neg_count = label_counts.get("NEGATIVE", 0)
        neu_count = label_counts.get("NEUTRAL",  0)

        # Confidence scores (exclude NEUTRAL rows — confidence is 0.0 for blanks
        # and unreliable for ambiguous reviews, so they'd skew the average)
        scored_rows = [r for r in game_rows if r["sentiment_label"] != "NEUTRAL"]
        if scored_rows:
            avg_conf = sum(float(r["sentiment_confidence"]) for r in scored_rows) / len(scored_rows)
        else:
            avg_conf = 0.0

        # Per-label confidence averages
        pos_rows = [r for r in game_rows if r["sentiment_label"] == "POSITIVE"]
        neg_rows = [r for r in game_rows if r["sentiment_label"] == "NEGATIVE"]

        avg_pos_conf = (
            sum(float(r["sentiment_confidence"]) for r in pos_rows) / len(pos_rows)
            if pos_rows else 0.0
        )
        avg_neg_conf = (
            sum(float(r["sentiment_confidence"]) for r in neg_rows) / len(neg_rows)
            if neg_rows else 0.0
        )

        # Word frequency — only English-ish reviews (non-empty text)
        pos_words = [
            w for r in pos_rows
            for w in tokenize(r["review_text"])
        ]
        neg_words = [
            w for r in neg_rows
            for w in tokenize(r["review_text"])
        ]

        games_stats[game_name] = {
            "total_reviews": total,
            "label_counts": {
                "POSITIVE": pos_count,
                "NEGATIVE": neg_count,
                "NEUTRAL":  neu_count,
            },
            "label_percentages": {
                "POSITIVE": round(pos_count / total * 100, 1),
                "NEGATIVE": round(neg_count / total * 100, 1),
                "NEUTRAL":  round(neu_count / total * 100, 1),
            },
            "avg_sentiment_confidence": round(avg_conf, 4),
            "avg_positive_confidence":  round(avg_pos_conf, 4),
            "avg_negative_confidence":  round(avg_neg_conf, 4),
            "top_words_positive": top_words(pos_words),
            "top_words_negative": top_words(neg_words),
        }

    # Overall totals across all games
    all_labels = Counter(r["sentiment_label"] for r in rows)
    all_scored = [r for r in rows if r["sentiment_label"] != "NEUTRAL"]
    overall_avg_conf = (
        sum(float(r["sentiment_confidence"]) for r in all_scored) / len(all_scored)
        if all_scored else 0.0
    )
    total_all = len(rows)

    all_pos_words = [
        w for r in rows if r["sentiment_label"] == "POSITIVE"
        for w in tokenize(r["review_text"])
    ]
    all_neg_words = [
        w for r in rows if r["sentiment_label"] == "NEGATIVE"
        for w in tokenize(r["review_text"])
    ]

    report = {
        "overall": {
            "total_reviews": total_all,
            "label_counts": dict(all_labels),
            "label_percentages": {
                lbl: round(cnt / total_all * 100, 1)
                for lbl, cnt in all_labels.items()
            },
            "avg_sentiment_confidence": round(overall_avg_conf, 4),
            "top_words_positive": top_words(all_pos_words),
            "top_words_negative": top_words(all_neg_words),
        },
        "by_game": games_stats,
    }

    return report

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_CSV}. Run sentiment_analysis.py first."
        )

    with INPUT_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info("Loaded %d rows from %s", len(rows), INPUT_CSV)

    report = build_report(rows)

    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Report saved → %s", OUTPUT_JSON)

    # Print a human-readable summary to stdout
    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    ov = report["overall"]
    print(f"  Total reviews : {ov['total_reviews']}")
    for lbl, pct in ov["label_percentages"].items():
        cnt = ov["label_counts"][lbl]
        print(f"  {lbl:<10}: {cnt:>5}  ({pct}%)")
    print(f"  Avg confidence: {ov['avg_sentiment_confidence']}")

    for game, stats in report["by_game"].items():
        print("\n" + "-" * 60)
        print(f"  {game}")
        print("-" * 60)
        print(f"  Total   : {stats['total_reviews']}")
        for lbl, pct in stats["label_percentages"].items():
            cnt = stats["label_counts"][lbl]
            print(f"  {lbl:<10}: {cnt:>4}  ({pct}%)")
        print(f"  Avg conf (all scored) : {stats['avg_sentiment_confidence']}")
        print(f"  Avg conf (positive)   : {stats['avg_positive_confidence']}")
        print(f"  Avg conf (negative)   : {stats['avg_negative_confidence']}")
        pos_top = ", ".join(f"{e['word']}({e['count']})" for e in stats["top_words_positive"])
        neg_top = ", ".join(f"{e['word']}({e['count']})" for e in stats["top_words_negative"])
        print(f"  Top positive words: {pos_top}")
        print(f"  Top negative words: {neg_top}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
