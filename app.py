import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Steam Review Analytics",
    page_icon="🎮",
    layout="wide",
)

REPORT_PATH = Path("output/sentiment_report.json")

# ---------------------------------------------------------------------------
# Load data (cached so it only reads from disk once)
# ---------------------------------------------------------------------------

@st.cache_data
def load_report() -> dict:
    if not REPORT_PATH.exists():
        st.error(
            f"Report file not found at `{REPORT_PATH}`. "
            "Run `analyze_sentiment.py` first."
        )
        st.stop()
    with REPORT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


report = load_report()
game_names = sorted(report["by_game"].keys())

# ---------------------------------------------------------------------------
# Sidebar — game selector
# ---------------------------------------------------------------------------

st.sidebar.title("🎮 Steam Review Analytics")
st.sidebar.markdown("Sentiment insights from 500 recent reviews per game.")

selected_game = st.sidebar.selectbox("Select a game", game_names)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** `distilbert-base-uncased-finetuned-sst-2-english`  \n"
    "**Reviews per game:** 500  \n"
    "**Neutral threshold:** confidence < 0.65"
)

# ---------------------------------------------------------------------------
# Pull stats for the selected game
# ---------------------------------------------------------------------------

stats    = report["by_game"][selected_game]
overall  = report["overall"]

pos_pct  = stats["label_percentages"]["POSITIVE"]
neg_pct  = stats["label_percentages"]["NEGATIVE"]
neu_pct  = stats["label_percentages"]["NEUTRAL"]
pos_cnt  = stats["label_counts"]["POSITIVE"]
neg_cnt  = stats["label_counts"]["NEGATIVE"]
neu_cnt  = stats["label_counts"]["NEUTRAL"]
total    = stats["total_reviews"]
avg_conf = stats["avg_sentiment_confidence"]
avg_pos  = stats["avg_positive_confidence"]
avg_neg  = stats["avg_negative_confidence"]

overall_pos_pct = overall["label_percentages"].get("POSITIVE", 0)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(f"📊 {selected_game}")
st.caption(f"Showing aggregated sentiment insights across {total} recent reviews.")
st.divider()

# ---------------------------------------------------------------------------
# Row 1 — Key metrics
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    label="Total Reviews",
    value=f"{total:,}",
)
col2.metric(
    label="✅ Positive",
    value=f"{pos_pct}%",
    delta=f"{pos_pct - overall_pos_pct:+.1f}% vs overall",
    delta_color="normal",
)
col3.metric(
    label="❌ Negative",
    value=f"{neg_pct}%",
)
col4.metric(
    label="➖ Neutral",
    value=f"{neu_pct}%",
)
col5.metric(
    label="Avg Confidence",
    value=f"{avg_conf:.1%}",
    help="Mean model confidence across all POSITIVE and NEGATIVE reviews.",
)

st.divider()

# ---------------------------------------------------------------------------
# Row 2 — Bar chart + confidence breakdown
# ---------------------------------------------------------------------------

chart_col, conf_col = st.columns([3, 2], gap="large")

with chart_col:
    st.subheader("Sentiment Breakdown")

    chart_df = pd.DataFrame(
        {
            "Sentiment": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
            "Reviews":   [pos_cnt,    neg_cnt,    neu_cnt],
        }
    ).set_index("Sentiment")

    st.bar_chart(
        chart_df,
        color=["#4CAF50"],   # single series → one colour; Streamlit applies it
        height=320,
    )

with conf_col:
    st.subheader("Confidence by Label")
    st.markdown(
        "Average model confidence tells you how *clearly* reviews skew "
        "positive or negative — values near 1.0 mean the model is very certain."
    )

    conf_df = pd.DataFrame(
        {
            "Label":      ["POSITIVE", "NEGATIVE"],
            "Avg Confidence": [avg_pos, avg_neg],
        }
    ).set_index("Label")

    st.bar_chart(conf_df, color=["#2196F3"], height=200)

    st.markdown(
        f"- **Positive avg:** `{avg_pos:.4f}`  \n"
        f"- **Negative avg:** `{avg_neg:.4f}`"
    )

st.divider()

# ---------------------------------------------------------------------------
# Row 3 — Top words
# ---------------------------------------------------------------------------

praise_col, complaint_col = st.columns(2, gap="large")

with praise_col:
    st.subheader("🟢 Top Praise Words")
    st.caption("Most frequent words in POSITIVE reviews (stopwords removed).")

    pos_words = stats["top_words_positive"]
    if pos_words:
        pos_df = (
            pd.DataFrame(pos_words)
            .rename(columns={"word": "Word", "count": "Mentions"})
            .set_index("Word")
        )
        st.bar_chart(pos_df, color=["#4CAF50"], height=320)

        # Also show as a ranked table for exact counts
        pos_df_display = pos_df.reset_index()
        pos_df_display.index = pos_df_display.index + 1   # rank from 1
        st.dataframe(pos_df_display, use_container_width=True, height=200)
    else:
        st.info("No positive word data available.")

with complaint_col:
    st.subheader("🔴 Top Complaint Words")
    st.caption("Most frequent words in NEGATIVE reviews (stopwords removed).")

    neg_words = stats["top_words_negative"]
    if neg_words:
        neg_df = (
            pd.DataFrame(neg_words)
            .rename(columns={"word": "Word", "count": "Mentions"})
            .set_index("Word")
        )
        st.bar_chart(neg_df, color=["#F44336"], height=320)

        neg_df_display = neg_df.reset_index()
        neg_df_display.index = neg_df_display.index + 1
        st.dataframe(neg_df_display, use_container_width=True, height=200)
    else:
        st.info("No negative word data available.")

st.divider()

# ---------------------------------------------------------------------------
# Row 4 — Cross-game comparison
# ---------------------------------------------------------------------------

st.subheader("📈 Cross-Game Comparison")
st.caption("Positive review percentage across all five games.")

comparison_rows = [
    {
        "Game":       name,
        "Positive %": data["label_percentages"]["POSITIVE"],
        "Negative %": data["label_percentages"]["NEGATIVE"],
        "Neutral %":  data["label_percentages"]["NEUTRAL"],
        "Avg Confidence": data["avg_sentiment_confidence"],
    }
    for name, data in sorted(report["by_game"].items())
]

comparison_df = pd.DataFrame(comparison_rows).set_index("Game")

tab1, tab2 = st.tabs(["Positive %", "Full Breakdown"])

with tab1:
    st.bar_chart(comparison_df[["Positive %"]], color=["#4CAF50"], height=300)

with tab2:
    st.bar_chart(comparison_df[["Positive %", "Negative %", "Neutral %"]], height=300)
    st.dataframe(comparison_df, use_container_width=True)
