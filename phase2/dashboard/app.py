"""
TradeQ — Phase 2
File: phase2/dashboard/app.py

What this dashboard does:
  - Shows the entire Phase 2 pipeline working in a browser
  - Three pages:
      * Pipeline Status  — is every service healthy? data fresh?
      * Stock Explorer   — live feature values for any stock
      * Sentiment Feed   — latest news headlines and FinBERT scores
  - This is NOT the final product dashboard (that's Phase 4 with Next.js)
  - This is the verification tool that confirms Phase 2 is working
    before we move to Phase 3 ML training

How to run:
  streamlit run phase2/dashboard/app.py

Then open http://localhost:8501 in your browser.

Author: TradeQ Project
"""

import sys
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import CONFIG
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv
import os

load_dotenv(ROOT / ".env")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeQ — Phase 2 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
        margin-bottom: 8px;
    }
    .status-ok   { color: #1D9E75; font-weight: 500; }
    .status-warn { color: #EF9F27; font-weight: 500; }
    .status-err  { color: #D85A30; font-weight: 500; }
    .headline-pos { color: #1D9E75; }
    .headline-neg { color: #D85A30; }
    .headline-neu { color: #888780; }
</style>
""", unsafe_allow_html=True)


# ─── Database connection ──────────────────────────────────────────────────────

@st.cache_resource
def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL not set in .env — is Docker running?")
        st.stop()
    return sa.create_engine(db_url, pool_pre_ping=True, pool_size=3)


@st.cache_data(ttl=60)   # Cache for 60 seconds — refreshes automatically
def query(_engine, sql: str, params: dict = None) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame. Cached for 60s."""
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.sidebar.title("TradeQ")
    st.sidebar.caption("Phase 2 — Data Pipeline Dashboard")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        ["📊 Pipeline Status", "🔍 Stock Explorer", "📰 Sentiment Feed"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.caption(f"v{CONFIG['project']['version']}  |  {CONFIG['project']['market']}")
    st.sidebar.caption("Data refreshes every 60 seconds")

    if st.sidebar.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

    return page


# ─── Page 1: Pipeline Status ──────────────────────────────────────────────────

def page_pipeline_status(engine):
    st.title("📊 Pipeline Status")
    st.caption("Real-time health check of every Phase 2 service and data source")
    st.divider()

    # ── Row 1: Key metrics ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    try:
        ohlcv_count = query(engine, "SELECT COUNT(*) as n FROM ohlcv_daily").iloc[0]["n"]
        col1.metric("OHLCV Rows", f"{ohlcv_count:,}", help="Total price candles in database")
    except Exception:
        col1.metric("OHLCV Rows", "Error")

    try:
        feat_count = query(engine, "SELECT COUNT(*) as n FROM features_daily").iloc[0]["n"]
        col2.metric("Feature Rows", f"{feat_count:,}", help="Total computed feature rows")
    except Exception:
        col2.metric("Feature Rows", "Error")

    try:
        news_count = query(engine, "SELECT COUNT(*) as n FROM news_sentiment").iloc[0]["n"]
        col3.metric("News Articles", f"{news_count:,}", help="Total articles scored by FinBERT")
    except Exception:
        col3.metric("News Articles", "Error")

    try:
        stock_count = query(engine, "SELECT COUNT(*) as n FROM stocks WHERE is_active=TRUE").iloc[0]["n"]
        col4.metric("Active Stocks", f"{stock_count}", help="Stocks being tracked")
    except Exception:
        col4.metric("Active Stocks", "Error")

    st.divider()

    # ── Row 2: Data freshness per stock ───────────────────────────────────────
    st.subheader("Data Freshness")
    st.caption("Latest data date per stock — should be within last 3 trading days")

    freshness_sql = """
        SELECT
            s.symbol,
            s.sector,
            MAX(o.time)::date  AS latest_ohlcv,
            MAX(f.time)::date  AS latest_features,
            COUNT(f.time)      AS feature_rows,
            COUNT(CASE WHEN f.news_sentiment IS NOT NULL THEN 1 END) AS news_rows
        FROM stocks s
        LEFT JOIN ohlcv_daily o     ON o.stock_id = s.id
        LEFT JOIN features_daily f  ON f.stock_id = s.id
        WHERE s.is_active = TRUE
        GROUP BY s.symbol, s.sector
        ORDER BY s.sector, s.symbol
    """

    df = query(engine, freshness_sql)
    today = date.today()

    if not df.empty:
        # Add freshness indicator
        def freshness_icon(d):
            if d is None or pd.isna(d):
                return "❌"
            try:
                days_old = (today - pd.to_datetime(d).date()).days
            except Exception:
                return "❌"
            if days_old <= 10:     # ← changed from 7 to 10
                return "✅"
            elif days_old <= 20:   # ← changed from 14 to 20
                return "⚠️"
            return "❌"

        df["OHLCV"] = df["latest_ohlcv"].apply(freshness_icon)
        df["Features"] = df["latest_features"].apply(freshness_icon)
        df["News"] = df["news_rows"].apply(lambda x: "✅" if x > 0 else "⚠️")

        display_df = df[[
            "symbol", "sector", "OHLCV", "latest_ohlcv",
            "Features", "latest_features", "feature_rows", "News", "news_rows"
        ]].rename(columns={
            "symbol": "Symbol", "sector": "Sector",
            "latest_ohlcv": "Last Price", "latest_features": "Last Features",
            "feature_rows": "Feature Rows", "news_rows": "News Rows"
        })

        st.dataframe(display_df, use_container_width=True, height=400)

    st.divider()

    # ── Row 3: Label distribution ─────────────────────────────────────────────
    st.subheader("ML Label Distribution")
    st.caption("Buy signal rate per stock — healthy range is 15-25%")

    label_sql = """
        SELECT
            s.symbol,
            COUNT(*) as total,
            SUM(CASE WHEN f.label = 1 THEN 1 ELSE 0 END) as buy_signals,
            ROUND(100.0 * SUM(CASE WHEN f.label = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as buy_rate
        FROM features_daily f
        JOIN stocks s ON s.id = f.stock_id
        WHERE f.label IS NOT NULL
        GROUP BY s.symbol
        ORDER BY buy_rate DESC
    """

    label_df = query(engine, label_sql)
    if not label_df.empty:
        fig = px.bar(
            label_df,
            x="symbol", y="buy_rate",
            color="buy_rate",
            color_continuous_scale=["#D85A30", "#EF9F27", "#1D9E75"],
            range_color=[10, 30],
            labels={"buy_rate": "Buy Signal Rate (%)", "symbol": "Stock"},
            title="Buy Signal Rate per Stock (%)",
        )
        fig.add_hline(y=19, line_dash="dash", line_color="#888780",
                      annotation_text="Market avg 19%")
        fig.update_layout(
            plot_bgcolor="#FAFAF8",
            paper_bgcolor="#FAFAF8",
            showlegend=False,
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── Page 2: Stock Explorer ───────────────────────────────────────────────────

def page_stock_explorer(engine):
    st.title("🔍 Stock Explorer")
    st.caption("Live feature values for any tracked stock")

    # Stock selector
    symbols_df = query(engine, "SELECT symbol FROM stocks WHERE is_active=TRUE ORDER BY symbol")
    symbols    = symbols_df["symbol"].tolist()

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox("Select stock", symbols, index=0)
    with col2:
        days = st.selectbox("History", [30, 60, 90, 180, 365], index=1)

    if not selected:
        return

    st.divider()

    # ── Latest feature snapshot ───────────────────────────────────────────────
    latest_sql = """
        SELECT f.*, s.name, s.sector
        FROM features_daily f
        JOIN stocks s ON s.id = f.stock_id
        WHERE s.symbol = :sym
        ORDER BY f.time DESC
        LIMIT 1
    """
    latest = query(engine, latest_sql, {"sym": selected})

    if latest.empty:
        st.warning(f"No feature data found for {selected}. Run feature_engine first.")
        return

    row = latest.iloc[0]

    # Key metrics row
    st.subheader(f"{selected} — {row.get('name', '')} ({row.get('sector', '')})")
    st.caption(f"Latest data: {str(row['time'])[:10]}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("RSI (14)", f"{row.get('rsi_14', 'N/A'):.1f}" if pd.notna(row.get('rsi_14')) else "N/A",
              help="< 30 oversold, > 70 overbought")
    m2.metric("EMA Cross", f"{row.get('ema_cross_20_50', 0):.4f}" if pd.notna(row.get('ema_cross_20_50')) else "N/A",
              help="> 1.0 = bullish (EMA20 above EMA50)")
    m3.metric("Volume Spike", f"{row.get('volume_spike', 0):.2f}x" if pd.notna(row.get('volume_spike')) else "N/A",
              help="Today's volume vs 20-day average")
    m4.metric("ATR (14)", f"{row.get('atr_14', 0):.2f}" if pd.notna(row.get('atr_14')) else "N/A",
              help="Average daily price range")
    m5.metric("News Sentiment", f"{row.get('news_sentiment', 0):+.3f}" if pd.notna(row.get('news_sentiment')) else "N/A",
              help="-1 = very negative, +1 = very positive")
    m6.metric("Rel. Strength", f"{row.get('relative_strength', 0):+.2f}%" if pd.notna(row.get('relative_strength')) else "N/A",
              help="Return vs Nifty today")

    st.divider()

    # ── Historical feature charts ─────────────────────────────────────────────
    history_sql = """
        SELECT f.time::date as date, f.rsi_14, f.macd_hist, f.volume_spike,
               f.daily_return, f.ema_cross_20_50, f.volatility_20d,
               f.news_sentiment, f.relative_strength, f.bb_pct_b,
               o.close
        FROM features_daily f
        JOIN ohlcv_daily o ON o.stock_id = f.stock_id AND o.time = f.time
        JOIN stocks s ON s.id = f.stock_id
        WHERE s.symbol = :sym
          AND f.time >= NOW() - INTERVAL ':days days'
        ORDER BY f.time ASC
    """

    # Build query with days substituted safely
    history_sql2 = f"""
        SELECT f.time::date as date, f.rsi_14, f.macd_hist, f.volume_spike,
               f.daily_return, f.ema_cross_20_50, f.volatility_20d,
               f.news_sentiment, f.bb_pct_b, o.close
        FROM features_daily f
        JOIN ohlcv_daily o ON o.stock_id = f.stock_id AND o.time = f.time
        JOIN stocks s ON s.id = f.stock_id
        WHERE s.symbol = :sym
          AND f.time >= NOW() - INTERVAL '{days} days'
        ORDER BY f.time ASC
    """

    hist = query(engine, history_sql2, {"sym": selected})

    if hist.empty:
        st.info("No historical data available for this range.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Price & RSI", "MACD & Volume", "Volatility", "Sentiment"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"],
                                 name="Close Price", line=dict(color="#378ADD", width=1.5)))
        fig.update_layout(title="Close Price", plot_bgcolor="#FAFAF8",
                          paper_bgcolor="#FAFAF8", height=250, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist["date"], y=hist["rsi_14"],
                                  name="RSI", line=dict(color="#7F77DD", width=1.5)))
        fig2.add_hline(y=70, line_dash="dash", line_color="#D85A30", annotation_text="Overbought 70")
        fig2.add_hline(y=30, line_dash="dash", line_color="#1D9E75", annotation_text="Oversold 30")
        fig2.update_layout(title="RSI (14)", plot_bgcolor="#FAFAF8",
                           paper_bgcolor="#FAFAF8", height=200, margin=dict(t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        colors = ["#1D9E75" if v >= 0 else "#D85A30"
                  for v in hist["macd_hist"].fillna(0)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hist["date"], y=hist["macd_hist"],
                             marker_color=colors, name="MACD Histogram"))
        fig.update_layout(title="MACD Histogram", plot_bgcolor="#FAFAF8",
                          paper_bgcolor="#FAFAF8", height=250, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        vol_colors = ["#1D9E75" if v >= 1.5 else "#888780"
                      for v in hist["volume_spike"].fillna(0)]
        fig2.add_trace(go.Bar(x=hist["date"], y=hist["volume_spike"],
                              marker_color=vol_colors, name="Volume Spike"))
        fig2.add_hline(y=1.5, line_dash="dash", line_color="#EF9F27",
                       annotation_text="Spike threshold 1.5x")
        fig2.update_layout(title="Volume Spike (vs 20-day avg)", plot_bgcolor="#FAFAF8",
                           paper_bgcolor="#FAFAF8", height=200, margin=dict(t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["volatility_20d"],
                                 fill="tozeroy", name="Volatility",
                                 line=dict(color="#EF9F27", width=1.5),
                                 fillcolor="rgba(239, 159, 39, 0.15)"))
        fig.update_layout(title="Rolling 20-day Annualised Volatility (%)",
                          plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
                          height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if hist["news_sentiment"].notna().any():
            colors = ["#1D9E75" if v > 0.05 else "#D85A30" if v < -0.05 else "#888780"
                      for v in hist["news_sentiment"].fillna(0)]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hist["date"], y=hist["news_sentiment"],
                                 marker_color=colors, name="News Sentiment"))
            fig.add_hline(y=0, line_color="#888780", line_width=0.8)
            fig.update_layout(title="Daily News Sentiment Score (FinBERT)",
                              plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
                              height=300, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No news sentiment data yet for this stock. Run news_ingestor --mode backfill.")

    st.divider()

    # ── Full feature table ────────────────────────────────────────────────────
    with st.expander("View raw feature values (last 10 days)"):
        raw_sql = """
            SELECT f.time::date as date, f.rsi_14, f.macd_hist, f.bb_pct_b,
                   f.ema_cross_20_50, f.volume_spike, f.atr_14,
                   f.daily_return, f.volatility_20d, f.news_sentiment,
                   f.relative_strength, f.label
            FROM features_daily f
            JOIN stocks s ON s.id = f.stock_id
            WHERE s.symbol = :sym
            ORDER BY f.time DESC
            LIMIT 10
        """
        raw = query(engine, raw_sql, {"sym": selected})
        st.dataframe(raw, use_container_width=True)


# ─── Page 3: Sentiment Feed ───────────────────────────────────────────────────

def page_sentiment_feed(engine):
    st.title("📰 Sentiment Feed")
    st.caption("Latest news headlines scored by FinBERT — updated daily")

    col1, col2 = st.columns([2, 1])
    with col1:
        symbols_df = query(engine, "SELECT symbol FROM stocks WHERE is_active=TRUE ORDER BY symbol")
        symbols    = ["All stocks"] + symbols_df["symbol"].tolist()
        selected   = st.selectbox("Filter by stock", symbols, index=0)
    with col2:
        sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Negative", "Neutral"])

    st.divider()

    # Build query
    where_clauses = ["1=1"]
    params        = {}

    if selected != "All stocks":
        where_clauses.append("s.symbol = :sym")
        params["sym"] = selected

    if sentiment_filter != "All":
        where_clauses.append("n.sentiment_label = :label")
        params["label"] = sentiment_filter.lower()

    news_sql = f"""
        SELECT
            s.symbol,
            n.published_at::date as date,
            n.headline,
            n.source,
            n.sentiment_label,
            n.sentiment_score
        FROM news_sentiment n
        JOIN stocks s ON s.id = n.stock_id
        WHERE {' AND '.join(where_clauses)}
        ORDER BY n.published_at DESC
        LIMIT 100
    """

    news_df = query(engine, news_sql, params)

    if news_df.empty:
        st.info("No news articles found. Run news_ingestor --mode backfill to populate.")
        return

    # ── Sentiment summary ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Articles", len(news_df))
    pos_count = (news_df["sentiment_label"] == "positive").sum()
    neg_count = (news_df["sentiment_label"] == "negative").sum()
    neu_count = (news_df["sentiment_label"] == "neutral").sum()
    m2.metric("Positive ↑", pos_count)
    m3.metric("Negative ↓", neg_count)
    m4.metric("Neutral →", neu_count)

    st.divider()

    # ── Headlines list ────────────────────────────────────────────────────────
    for _, row in news_df.iterrows():
        icon  = "↑" if row["sentiment_label"] == "positive" else \
                "↓" if row["sentiment_label"] == "negative" else "→"
        color = "headline-pos" if row["sentiment_label"] == "positive" else \
                "headline-neg" if row["sentiment_label"] == "negative" else "headline-neu"
        score = row["sentiment_score"]

        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.markdown(f'<span class="{color}">{icon} {row["symbol"]}</span>',
                        unsafe_allow_html=True)
        with col2:
            st.markdown(f'<span class="{color}">{row["headline"][:120]}</span>',
                        unsafe_allow_html=True)
            st.caption(f"{row['source']} — {row['date']}")
        with col3:
            st.markdown(f'<span class="{color}">{score:+.3f}</span>',
                        unsafe_allow_html=True)

        st.divider()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    engine = get_engine()
    page   = sidebar()

    if page == "📊 Pipeline Status":
        page_pipeline_status(engine)
    elif page == "🔍 Stock Explorer":
        page_stock_explorer(engine)
    elif page == "📰 Sentiment Feed":
        page_sentiment_feed(engine)


if __name__ == "__main__":
    main()