"""
TradeQ — Phase 2
File: phase2/services/news_ingestor/ingestor.py

What this service does:
  - Fetches financial news headlines for every Nifty 50 stock from NewsAPI
  - Runs each headline through FinBERT — a financial-domain NLP model
  - FinBERT classifies each headline as positive / negative / neutral
    with a confidence score
  - Stores every article in news_sentiment table
  - Aggregates daily sentiment scores and writes them into features_daily
    (news_sentiment, news_count, news_positive, news_negative columns)
  - Safe to re-run — skips already-processed articles

Modes:
  fetch    — fetch today's news for all stocks and score sentiment
  backfill — fetch last 30 days of news (NewsAPI free tier limit)
  symbol   — fetch news for one specific stock
  aggregate— roll up news_sentiment table into features_daily columns
  verify   — show summary of what is in news_sentiment table

How to run:
  python3 phase2/services/news_ingestor/ingestor.py --mode fetch
  python3 phase2/services/news_ingestor/ingestor.py --mode backfill
  python3 phase2/services/news_ingestor/ingestor.py --mode symbol --symbol RELIANCE
  python3 phase2/services/news_ingestor/ingestor.py --mode aggregate
  python3 phase2/services/news_ingestor/ingestor.py --mode verify

Author: TradeQ Project
"""

import sys
import time
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, date, timedelta

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import CONFIG, get_symbols
import requests
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv
import os

load_dotenv(ROOT / ".env")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── NewsAPI config ───────────────────────────────────────────────────────────
NEWS_API_KEY  = os.getenv("NEWS_API_KEY")
NEWS_API_URL  = "https://newsapi.org/v2/everything"
# Free tier: 100 requests/day, articles up to 30 days old
MAX_ARTICLES_PER_STOCK = 10    # Keep it low to stay within free tier limits
NEWS_LANGUAGE          = "en"
NEWS_SORT_BY           = "publishedAt"

# ─── Company name mappings ────────────────────────────────────────────────────
# NewsAPI searches by company name, not ticker symbol.
# These are the search terms that return the most relevant Indian financial news.
SEARCH_TERMS = {
    "RELIANCE.NS":   "Reliance Industries",
    "TCS.NS":        "Tata Consultancy Services TCS",
    "HDFCBANK.NS":   "HDFC Bank",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ICICIBANK.NS":  "ICICI Bank",
    "INFOSYS.NS":    "Infosys",
    "SBIN.NS":       "State Bank of India SBI",
    "HINDUNILVR.NS": "Hindustan Unilever HUL",
    "ITC.NS":        "ITC Limited",
    "LT.NS":         "Larsen Toubro",
    "KOTAKBANK.NS":  "Kotak Mahindra Bank",
    "AXISBANK.NS":   "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "MARUTI.NS":     "Maruti Suzuki",
    "HCLTECH.NS":    "HCL Technologies",
    "ASIANPAINT.NS": "Asian Paints",
    "ADANIENT.NS":   "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "TITAN.NS":      "Titan Company",
    "WIPRO.NS":      "Wipro",
    "NTPC.NS":       "NTPC Limited",
    "POWERGRID.NS":  "Power Grid India",
    "COALINDIA.NS":  "Coal India",
    "SUNPHARMA.NS":  "Sun Pharmaceutical",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "ONGC.NS":       "ONGC Oil Natural Gas",
    "M&M.NS":        "Mahindra Mahindra",
    "JSWSTEEL.NS":   "JSW Steel",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS":  "Tata Steel",
    "HINDALCO.NS":   "Hindalco Industries",
    "TECHM.NS":      "Tech Mahindra",
    "INDUSINDBK.NS": "IndusInd Bank",
    "CIPLA.NS":      "Cipla",
    "GRASIM.NS":     "Grasim Industries",
    "BRITANNIA.NS":  "Britannia Industries",
    "DRREDDY.NS":    "Dr Reddys Laboratories",
    "DIVISLAB.NS":   "Divis Laboratories",
    "BPCL.NS":       "Bharat Petroleum BPCL",
    "TATACONSUM.NS": "Tata Consumer Products",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "EICHERMOT.NS":  "Eicher Motors Royal Enfield",
    "NESTLEIND.NS":  "Nestle India",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "SBILIFE.NS":    "SBI Life Insurance",
    "HDFCLIFE.NS":   "HDFC Life Insurance",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "TRENT.NS":      "Trent Westside Zudio",
}


# ─── FinBERT sentiment model ──────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Wraps FinBERT for financial sentiment classification.
    Loads the model once and reuses it for all headlines.
    Lazy-loaded on first use to avoid slow startup when just verifying.
    """
    def __init__(self):
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        log.info("Loading FinBERT model (first run downloads ~438MB, cached after)...")
        from transformers import pipeline
        self._pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        log.info("FinBERT loaded successfully.")

    def score(self, text: str) -> tuple[str, float]:
        """
        Score a single headline.

        Args:
            text: Headline string

        Returns:
            Tuple of (label, score) where:
              label: "positive" | "negative" | "neutral"
              score: confidence 0.0-1.0, signed:
                     positive → +score, negative → -score, neutral → 0
        """
        self._load()
        try:
            text = text.strip()[:512]   # FinBERT max length
            if not text:
                return "neutral", 0.0

            result = self._pipeline(text)[0]
            label  = result["label"].lower()     # "positive" / "negative" / "neutral"
            conf   = result["score"]

            # Convert to signed score: positive=+conf, negative=-conf, neutral=0
            if label == "positive":
                signed = round(conf, 3)
            elif label == "negative":
                signed = round(-conf, 3)
            else:
                signed = 0.0

            return label, signed

        except Exception as e:
            log.warning(f"FinBERT scoring failed for '{text[:50]}...': {e}")
            return "neutral", 0.0

    def score_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Score a list of headlines. More efficient than calling score() in a loop."""
        return [self.score(t) for t in texts]


# Singleton — load once, reuse everywhere
analyzer = SentimentAnalyzer()


# ─── Database ─────────────────────────────────────────────────────────────────

def get_engine() -> sa.Engine:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL not set in .env")
    return sa.create_engine(db_url, pool_pre_ping=True, pool_size=5)


def get_stock_map(engine: sa.Engine) -> dict[str, int]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT symbol, id FROM stocks WHERE is_active = TRUE")
        ).fetchall()
    return {row.symbol: row.id for row in rows}


def get_existing_urls(engine: sa.Engine, stock_id: int) -> set[str]:
    """Return URLs already stored for this stock — used to skip duplicates."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT url FROM news_sentiment WHERE stock_id = :sid AND url IS NOT NULL"),
            {"sid": stock_id}
        ).fetchall()
    return {row.url for row in rows}


# ─── NewsAPI fetching ─────────────────────────────────────────────────────────

def fetch_news(symbol: str, from_date: date, to_date: date) -> list[dict]:
    """
    Fetch news articles for a stock from NewsAPI.

    Args:
        symbol:    NSE ticker e.g. "RELIANCE.NS"
        from_date: Start date for article search
        to_date:   End date for article search

    Returns:
        List of article dicts with keys: title, source, url, publishedAt
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "your_key_here":
        log.error("NEWS_API_KEY not set in .env — get a free key at newsapi.org")
        return []

    search_term = SEARCH_TERMS.get(symbol, symbol.replace(".NS", ""))

    params = {
        "q":        f'"{search_term}" stock OR shares OR NSE OR BSE OR market',
        "from":     from_date.isoformat(),
        "to":       to_date.isoformat(),
        "language": NEWS_LANGUAGE,
        "sortBy":   NEWS_SORT_BY,
        "pageSize": MAX_ARTICLES_PER_STOCK,
        "apiKey":   NEWS_API_KEY,
    }

    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=15)

        if resp.status_code == 429:
            log.warning("NewsAPI rate limit hit — pausing 60 seconds")
            time.sleep(60)
            return []

        if resp.status_code != 200:
            log.warning(f"NewsAPI returned {resp.status_code} for {symbol}: {resp.text[:100]}")
            return []

        data     = resp.json()
        articles = data.get("articles", [])
        return articles

    except requests.exceptions.Timeout:
        log.warning(f"NewsAPI timeout for {symbol}")
        return []
    except Exception as e:
        log.error(f"NewsAPI fetch failed for {symbol}: {e}")
        return []


# ─── Insert news records ──────────────────────────────────────────────────────

INSERT_NEWS_SQL = text("""
    INSERT INTO news_sentiment
        (stock_id, published_at, fetched_at, headline, source, url,
         sentiment_label, sentiment_score, is_processed)
    VALUES
        (:stock_id, :published_at, NOW(), :headline, :source, :url,
         :sentiment_label, :sentiment_score, FALSE)
    ON CONFLICT DO NOTHING
""")


def process_and_store(
    engine: sa.Engine,
    stock_id: int,
    symbol: str,
    articles: list[dict],
    existing_urls: set[str],
) -> tuple[int, int, int]:
    """
    Score articles with FinBERT and insert into news_sentiment.

    Returns:
        Tuple of (inserted, positive_count, negative_count)
    """
    inserted   = 0
    n_positive = 0
    n_negative = 0

    rows = []
    for article in articles:
        url = article.get("url", "")

        # Skip duplicates
        if url in existing_urls:
            continue

        headline = article.get("title") or article.get("description") or ""
        headline = headline.strip()
        if not headline or headline == "[Removed]":
            continue

        # Score with FinBERT
        label, score = analyzer.score(headline)

        published_raw = article.get("publishedAt", "")
        try:
            published_at = datetime.fromisoformat(
                published_raw.replace("Z", "+00:00")
            )
        except Exception:
            published_at = datetime.utcnow()

        source = article.get("source", {}).get("name", "unknown")

        rows.append({
            "stock_id":        stock_id,
            "published_at":    published_at,
            "headline":        headline[:500],   # DB column limit
            "source":          source[:100],
            "url":             url[:500] if url else None,
            "sentiment_label": label,
            "sentiment_score": score,
        })

        if label == "positive":
            n_positive += 1
        elif label == "negative":
            n_negative += 1

        existing_urls.add(url)

    if rows:
        with engine.begin() as conn:
            result = conn.execute(INSERT_NEWS_SQL, rows)
            inserted = result.rowcount

    return inserted, n_positive, n_negative


# ─── Aggregate into features_daily ───────────────────────────────────────────

def aggregate_sentiment(engine: sa.Engine) -> None:
    """
    Roll up news_sentiment into features_daily.

    For each stock × date:
      - Compute average sentiment score across all articles that day
      - Count total articles, positive articles, negative articles
      - Write into the corresponding features_daily row

    Only processes unprocessed articles (is_processed = FALSE).
    """
    log.info("Aggregating news sentiment into features_daily...")

    agg_sql = text("""
        SELECT
            stock_id,
            published_at::date AS news_date,
            COUNT(*)                              AS article_count,
            ROUND(AVG(sentiment_score)::numeric, 3) AS avg_sentiment,
            SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) AS pos_count,
            SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) AS neg_count
        FROM news_sentiment
        WHERE is_processed = FALSE
        GROUP BY stock_id, published_at::date
    """)

    update_sql = text("""
        UPDATE features_daily SET
            news_sentiment = :avg_sentiment,
            news_count     = :article_count,
            news_positive  = :pos_count,
            news_negative  = :neg_count,
            updated_at     = NOW()
        WHERE stock_id = :stock_id
          AND time::date = :news_date
    """)

    mark_processed_sql = text("""
        UPDATE news_sentiment
        SET is_processed = TRUE
        WHERE is_processed = FALSE
          AND published_at::date = :news_date
          AND stock_id = :stock_id
    """)

    with engine.connect() as conn:
        agg_rows = conn.execute(agg_sql).fetchall()

    if not agg_rows:
        log.info("No unprocessed news articles found.")
        return

    updated = 0
    with engine.begin() as conn:
        for row in agg_rows:
            result = conn.execute(update_sql, {
                "stock_id":      row.stock_id,
                "news_date":     row.news_date,
                "avg_sentiment": float(row.avg_sentiment),
                "article_count": int(row.article_count),
                "pos_count":     int(row.pos_count),
                "neg_count":     int(row.neg_count),
            })
            if result.rowcount > 0:
                updated += 1
                conn.execute(mark_processed_sql, {
                    "stock_id":  row.stock_id,
                    "news_date": row.news_date,
                })

    log.info(f"Aggregation complete — {updated} features_daily rows updated with sentiment")


# ─── Modes ────────────────────────────────────────────────────────────────────

def run_fetch(engine: sa.Engine, stock_map: dict[str, int], days_back: int = 1) -> None:
    """
    Fetch news for all stocks for the last `days_back` days.
    Default is 1 day — designed to run daily.
    """
    to_date   = date.today()
    from_date = to_date - timedelta(days=days_back)

    log.info(f"Fetching news for {len(stock_map)} stocks | {from_date} → {to_date}")
    log.info("─" * 55)

    total_inserted  = 0
    total_articles  = 0
    api_calls       = 0

    for i, (symbol, stock_id) in enumerate(stock_map.items(), 1):
        existing_urls = get_existing_urls(engine, stock_id)
        articles      = fetch_news(symbol, from_date, to_date)
        api_calls    += 1

        if not articles:
            log.info(f"[{i:>2}/{len(stock_map)}] {symbol:<20} — no articles found")
        else:
            inserted, n_pos, n_neg = process_and_store(
                engine, stock_id, symbol, articles, existing_urls
            )
            total_inserted += inserted
            total_articles += len(articles)
            log.info(
                f"[{i:>2}/{len(stock_map)}] {symbol:<20} "
                f"{len(articles):>3} articles  "
                f"{inserted:>3} new  "
                f"pos={n_pos} neg={n_neg}"
            )

        # NewsAPI free tier: 100 requests/day — pace ourselves
        time.sleep(1.2)

    print("\n" + "═" * 55)
    print(f"  FETCH SUMMARY")
    print("═" * 55)
    print(f"  API calls made   : {api_calls}")
    print(f"  Articles fetched : {total_articles}")
    print(f"  New rows stored  : {total_inserted}")
    print("═" * 55 + "\n")

    # Auto-aggregate after fetching
    aggregate_sentiment(engine)


def run_backfill(engine: sa.Engine, stock_map: dict[str, int]) -> None:
    """
    Fetch last 30 days of news (max NewsAPI free tier allows).
    Run once to populate historical sentiment.
    Note: Free tier gives 100 req/day. With 50 stocks this uses all 100.
    Run in two batches of 25 if you hit the limit.
    """
    log.info("MODE: backfill — fetching 30 days of news for all stocks")
    log.info("Note: This uses ~50 of your 100 daily NewsAPI requests")
    log.info("─" * 55)
    run_fetch(engine, stock_map, days_back=30)


def run_single_symbol(
    engine: sa.Engine, stock_map: dict[str, int], symbol: str, days_back: int = 7
) -> None:
    """Fetch and score news for one stock."""
    symbol = symbol.upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"

    if symbol not in stock_map:
        log.error(f"{symbol} not found in stocks table")
        return

    stock_id      = stock_map[symbol]
    existing_urls = get_existing_urls(engine, stock_id)
    to_date       = date.today()
    from_date     = to_date - timedelta(days=days_back)

    log.info(f"Fetching {days_back} days of news for {symbol}")
    articles = fetch_news(symbol, from_date, to_date)

    if not articles:
        log.info("No articles found")
        return

    inserted, n_pos, n_neg = process_and_store(
        engine, stock_id, symbol, articles, existing_urls
    )

    log.info(f"Done — {len(articles)} fetched, {inserted} stored | pos={n_pos} neg={n_neg}")

    # Show the headlines and scores
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT published_at::date as dt, headline, sentiment_label, sentiment_score
            FROM news_sentiment
            WHERE stock_id = :sid
            ORDER BY published_at DESC
            LIMIT 10
        """), {"sid": stock_id}).fetchall()

    print(f"\n  Latest headlines for {symbol}:")
    print("─" * 70)
    for row in rows:
        icon = "↑" if row.sentiment_label == "positive" else ("↓" if row.sentiment_label == "negative" else "→")
        print(f"  {icon} [{row.dt}] {row.headline[:60]:<60} ({row.sentiment_score:+.3f})")
    print()

    aggregate_sentiment(engine)


def run_verify(engine: sa.Engine) -> None:
    """Show what is currently in news_sentiment and how it maps to features_daily."""
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM news_sentiment")
        ).scalar()

        label_dist = conn.execute(text("""
            SELECT sentiment_label, COUNT(*) as cnt,
                   ROUND(AVG(sentiment_score)::numeric, 3) as avg_score
            FROM news_sentiment
            GROUP BY sentiment_label ORDER BY cnt DESC
        """)).fetchall()

        per_stock = conn.execute(text("""
            SELECT s.symbol,
                   COUNT(*) as articles,
                   ROUND(AVG(n.sentiment_score)::numeric, 3) as avg_sentiment,
                   MAX(n.published_at)::date as latest
            FROM news_sentiment n
            JOIN stocks s ON s.id = n.stock_id
            GROUP BY s.symbol
            ORDER BY articles DESC
            LIMIT 20
        """)).fetchall()

        features_with_news = conn.execute(text("""
            SELECT COUNT(*) FROM features_daily
            WHERE news_sentiment IS NOT NULL
        """)).scalar()

    print("\n" + "═" * 60)
    print(f"  NEWS SENTIMENT SUMMARY")
    print("═" * 60)
    print(f"  Total articles stored     : {total:,}")
    print(f"  Features rows with news   : {features_with_news:,}")
    print()

    if label_dist:
        print("  Sentiment distribution:")
        for row in label_dist:
            icon = "↑" if row.sentiment_label == "positive" else ("↓" if row.sentiment_label == "negative" else "→")
            print(f"    {icon} {row.sentiment_label:<10} {row.cnt:>6} articles  avg score: {row.avg_score:+.3f}")

    if per_stock:
        print(f"\n  Top stocks by article count:")
        print(f"  {'Symbol':<20} {'Articles':>9}  {'Avg Sentiment':>14}  {'Latest'}")
        print(f"  {'─'*20} {'─'*9}  {'─'*14}  {'─'*12}")
        for row in per_stock:
            icon = "↑" if row.avg_sentiment > 0.05 else ("↓" if row.avg_sentiment < -0.05 else "→")
            print(f"  {row.symbol:<20} {row.articles:>9}  {icon} {row.avg_sentiment:>+12.3f}  {row.latest}")

    print("═" * 60 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TradeQ news ingestor — fetches headlines and scores sentiment with FinBERT"
    )
    parser.add_argument(
        "--mode",
        choices=["fetch", "backfill", "symbol", "aggregate", "verify"],
        default="fetch",
    )
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Days back to fetch news (default 1 for daily mode)"
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    engine = get_engine()

    log.info("TradeQ — News Ingestor")
    log.info(f"Mode: {args.mode}")
    log.info("─" * 55)

    if args.mode == "verify":
        run_verify(engine)
        return

    if args.mode == "aggregate":
        aggregate_sentiment(engine)
        return

    stock_map = get_stock_map(engine)
    log.info(f"Active stocks: {len(stock_map)}")

    start = time.time()

    if args.mode == "fetch":
        run_fetch(engine, stock_map, days_back=args.days)

    elif args.mode == "backfill":
        run_backfill(engine, stock_map)

    elif args.mode == "symbol":
        if not args.symbol:
            log.error("--symbol required for mode=symbol")
            sys.exit(1)
        run_single_symbol(engine, stock_map, args.symbol, days_back=args.days)

    log.info(f"Done in {round(time.time() - start, 1)}s")


if __name__ == "__main__":
    main()