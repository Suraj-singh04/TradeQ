# TradeQ

AI-powered daily stock signal engine for the Indian equity market (NSE/BSE). Ingests price history, computes 30+ technical and macro features, scores financial news sentiment with FinBERT, and ranks Nifty 50 stocks by predicted next-session outperformance probability.

---

## Architecture

```
External Sources
├── yfinance          → OHLCV price data (free, no key required)
├── NewsAPI           → Financial headlines (100 req/day free tier)
└── HuggingFace       → ProsusAI/finbert model weights (downloaded once)

Ingestion Layer
├── price_ingestor    → Downloads OHLCV, writes to ohlcv_daily
├── feature_engine    → Reads OHLCV, computes indicators, writes to features_daily
└── news_ingestor     → Fetches headlines, runs FinBERT, aggregates into features_daily

Storage Layer
└── PostgreSQL 15 + TimescaleDB (Docker)
    ├── stocks             Reference table — 50 Nifty companies
    ├── ohlcv_daily        TimescaleDB hypertable — 1-month partitions
    ├── features_daily     TimescaleDB hypertable — 3-month partitions
    ├── news_sentiment     Raw article store with FinBERT scores
    └── predictions        Model output (Phase 3)

Presentation Layer
├── Streamlit dashboard    Phase 2 verification UI (localhost:8501)
└── FastAPI + Next.js      Phase 4 production dashboard (planned)
```

---

## Data Flow

```
yfinance
  └─→ price_ingestor ──────────────────→ ohlcv_daily
                                              │
                                        feature_engine
                                         (pandas-ta)
                                              │
                                        features_daily ←── news_ingestor
                                         (30+ cols)          (FinBERT)
                                              │
                                        prediction_engine      [Phase 3]
                                         (XGBoost model)
                                              │
                                        predictions table
                                              │
                                        FastAPI /predictions   [Phase 4]
                                              │
                                        Next.js Dashboard      [Phase 4]
```

---

## Feature Pipeline

The feature engine reads raw OHLCV from the database and computes the following feature groups per stock per trading day:

**Technical Indicators (20 features)**
RSI-14, MACD(12,26,9) + histogram + signal, Bollinger Bands (%B, width, upper, lower),
EMA-20, EMA-50, EMA-200, EMA cross ratio, ATR-14, ADX-14, OBV, OBV slope,
Stochastic %K/%D, CCI-20, Williams %R, VWAP deviation

**Price Action (10 features)**
Daily return, log return, body percentage, gap percentage, close position in daily range,
52-week high distance, 52-week low distance, rolling 20-day volatility, consecutive
green streak, consecutive red streak

**Volume (3 features)**
20-day volume moving average, volume spike ratio (today / MA20), OBV 5-day slope

**Macro Context (5 features)**
Nifty 50 daily return, India VIX level, USD/INR daily change, crude oil daily change,
relative strength (stock return minus Nifty return)

**News Sentiment (4 features)**
FinBERT composite score (-1 to +1), total article count, positive article count,
negative article count

**ML Target Label**
`label = 1` if stock outperformed Nifty 50 by more than 1% the following session,
`label = 0` otherwise. Last row always NULL (future unknown).

Current label distribution across 56,824 labelled rows: 19% buy signal, 81% hold.

---

## Service Interactions

```
config.py
  └─ imported by every service — single source of truth for:
     symbols, paths, indicator parameters, ML hyperparameters, market schedule

price_ingestor
  ├─ reads:  stocks table (symbol → stock_id mapping)
  ├─ writes: ohlcv_daily
  └─ modes:  backfill | daily | symbol | migrate | verify

feature_engine
  ├─ reads:  ohlcv_daily (per stock, full history)
  ├─ reads:  yfinance (macro indices — live download per run)
  ├─ writes: features_daily
  └─ modes:  full | daily | symbol | labels | verify

news_ingestor
  ├─ reads:  stocks table (symbol → stock_id mapping)
  ├─ reads:  NewsAPI (headlines per company name search term)
  ├─ reads:  HuggingFace (ProsusAI/finbert — lazy loaded)
  ├─ writes: news_sentiment (raw articles + scores)
  ├─ writes: features_daily (aggregated sentiment columns)
  └─ modes:  fetch | backfill | symbol | aggregate | verify

seed_stocks.py
  ├─ reads:  yfinance (company name, market cap)
  └─ writes: stocks (upsert — safe to re-run)

Streamlit dashboard
  └─ reads:  all tables (read-only, @st.cache_data TTL=60s)
```

---

## Database Schema

```sql
stocks              (id, symbol, name, sector, industry, market_cap_cr, is_active)
ohlcv_daily         (time, stock_id, open, high, low, close, volume, adjusted_close)
                    → TimescaleDB hypertable, partitioned by month
features_daily      (time, stock_id, rsi_14, macd, macd_signal, macd_hist,
                     bb_upper, bb_lower, bb_pct_b, bb_width, ema_20, ema_50, ema_200,
                     ema_cross_20_50, atr_14, adx_14, obv, obv_slope, vwap_dev,
                     stoch_k, stoch_d, cci_20, williams_r, daily_return, log_return,
                     body_pct, gap_pct, high_52w_dist, low_52w_dist, close_position,
                     consec_green, consec_red, volatility_20d, volume_spike,
                     volume_ma_20, nifty_daily_ret, india_vix, relative_strength,
                     usd_inr_change, crude_change, news_sentiment, news_count,
                     news_positive, news_negative, label, next_day_return)
                    → TimescaleDB hypertable, partitioned by quarter
news_sentiment      (id, stock_id, published_at, headline, source, url,
                     sentiment_label, sentiment_score, is_processed)
predictions         (id, prediction_date, stock_id, rank, confidence_score,
                     model_version, top_factors, actual_return, beat_nifty)

Views:
  v_latest_features      → most recent feature row per stock (used by API)
  v_today_predictions    → today's ranked predictions joined with key features
```

---

## Phase Connections

```
Phase 1  →  Phase 2
  data/raw/*.csv (Phase 1 output) ingested into ohlcv_daily via
  price_ingestor --mode migrate. All Phase 1 indicator logic
  reimplemented inside feature_engine using the same config.py
  parameters, ensuring consistency.

Phase 2  →  Phase 3
  features_daily is the direct training input for Phase 3 XGBoost model.
  The label column (computed by feature_engine) is the target variable.
  XGBoost hyperparameters are pre-defined in CONFIG['ml']['xgb_params'].
  The predictions table (written by Phase 3) is already created in the
  Phase 2 schema.

Phase 3  →  Phase 4
  prediction_engine service writes daily ranked predictions to the
  predictions table. FastAPI api-gateway (Phase 4) reads from
  v_today_predictions view and serves /predictions/today endpoint.
  Next.js frontend consumes this endpoint for the morning scan screen.
```

---

## ML/AI Components

**FinBERT (Phase 2 — active)**
Model: `ProsusAI/finbert` via HuggingFace transformers pipeline.
Input: financial headline string (truncated to 512 tokens).
Output: label (positive/negative/neutral) + confidence score.
Signed score conversion: positive → +score, negative → -score, neutral → 0.
Loaded lazily on first call to SentimentAnalyzer.score(). Cached locally
after first download (~438 MB).

**XGBoost Classifier (Phase 3 — planned)**
Target: binary classification — did stock beat Nifty by >1% next session?
Features: all non-null columns in features_daily (~30 available currently).
Validation: walk-forward cross-validation (5 folds, no random shuffle).
Hyperparameters defined in config.py:
  n_estimators=500, max_depth=6, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, eval_metric=logloss.
Tracking: MLflow experiment logging.
Retraining: weekly, Sunday 02:00 IST.

---

## Dashboard Flow

```
Streamlit dashboard (phase2/dashboard/app.py)

Page 1: Pipeline Status
  SQL → stocks + ohlcv_daily + features_daily + news_sentiment counts
  SQL → per-stock freshness (MAX(time) per table per stock)
  SQL → label distribution per stock (bar chart via plotly)

Page 2: Stock Explorer
  SQL → latest feature snapshot for selected stock (v_latest_features)
  SQL → historical features + OHLCV joined for chart range
  Tabs: Close + RSI | MACD histogram + Volume spike |
        Rolling volatility | News sentiment timeline

Page 3: Sentiment Feed
  SQL → news_sentiment filtered by stock and/or sentiment label
  Renders: signed score, headline, source, date per article

All queries: @st.cache_data(ttl=60) — auto-refresh every 60 seconds.
DB connection: @st.cache_resource — single SQLAlchemy pool reused.
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Database | PostgreSQL 15 + TimescaleDB |
| Containers | Docker + Docker Compose |
| ORM | SQLAlchemy 2.x |
| Price data | yfinance |
| Indicators | pandas-ta |
| News | NewsAPI (free tier) |
| Sentiment NLP | ProsusAI/finbert (HuggingFace transformers) |
| ML framework | XGBoost + scikit-learn (Phase 3) |
| Verification UI | Streamlit + Plotly |
| Production API | FastAPI (Phase 4) |
| Production UI | Next.js + TailwindCSS (Phase 4) |
| Scheduling | Celery + Redis (Phase 4) |
| ML tracking | MLflow (Phase 3) |

---

## Folder Structure

```
TradeQ/
├── config.py                              Central config — all settings
├── utils.py                               Shared helpers
├── docker-compose.yml                     PostgreSQL + TimescaleDB container
├── .env                                   Secrets — never commit
├── requirements.txt                       pip dependencies
│
├── phase1/                                EDA scripts (CSV-based, no DB required)
│   ├── download_stocks.py                 Fetch OHLCV to data/raw/
│   ├── plot_stock.py                      Candlestick chart generator
│   ├── explore_stock.py                   7-panel statistical analysis
│   └── compare_stocks.py                 Multi-stock Sharpe ranking
│
├── phase2/
│   ├── database/
│   │   ├── migrations/001_initial_schema.sql     Full DB schema
│   │   └── seeds/seed_stocks.py                 Populate stocks table
│   ├── services/
│   │   ├── price_ingestor/ingestor.py           OHLCV → ohlcv_daily
│   │   ├── feature_engine/engine.py             Indicators → features_daily
│   │   └── news_ingestor/ingestor.py            FinBERT → news_sentiment
│   └── dashboard/app.py                         Streamlit verification UI
│
├── data/
│   ├── raw/       Phase 1 CSVs (gitignored)
│   ├── charts/    Generated PNG charts (gitignored)
│   ├── models/    Trained model files (Phase 3)
│   └── logs/      Service logs
│
├── phase3/        ML training pipeline (planned)
└── phase4/        FastAPI + Next.js (planned)
```

---

## Setup

### Prerequisites

- Python 3.12
- Docker + Docker Compose
- ~2 GB free disk (FinBERT model cache + database volume)
- NewsAPI key — free at newsapi.org

### Install

```bash
git clone https://github.com/<username>/TradeQ.git
cd TradeQ
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment

```bash
cp .env.example .env
# Edit .env and set NEWS_API_KEY
```

### Database

```bash
docker-compose up -d
docker-compose ps   # confirm status: Up

# Apply schema (only needed if volume was wiped)
docker exec -i tradeq_postgres psql -U tradeq -d tradeq_db \
  < phase2/database/migrations/001_initial_schema.sql

# Seed stock metadata
python3 phase2/database/seeds/seed_stocks.py
```

---

## Running Services

```bash
source venv/bin/activate
docker-compose start

# Initial population (run once)
python3 phase2/services/price_ingestor/ingestor.py --mode backfill
python3 phase2/services/feature_engine/engine.py --mode full
python3 phase2/services/news_ingestor/ingestor.py --mode backfill

# Daily update (run each morning before 09:15 IST)
python3 phase2/services/price_ingestor/ingestor.py --mode daily
python3 phase2/services/feature_engine/engine.py --mode daily
python3 phase2/services/news_ingestor/ingestor.py --mode fetch

# Verification dashboard
streamlit run phase2/dashboard/app.py
# Open http://localhost:8501
```

---

## Phase 1 Tools (no Docker required)

```bash
python3 phase1/download_stocks.py
python3 phase1/plot_stock.py RELIANCE
python3 phase1/explore_stock.py HDFCBANK
python3 phase1/compare_stocks.py
```

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| POSTGRES_USER | Yes | Database user |
| POSTGRES_PASSWORD | Yes | Database password |
| POSTGRES_DB | Yes | Database name |
| DATABASE_URL | Yes | Full SQLAlchemy connection string |
| NEWS_API_KEY | Yes | NewsAPI.org free tier key |
| ENV | No | development / production |

---

## Troubleshooting

| Symptom | Resolution |
|---|---|
| `ModuleNotFoundError` | `source venv/bin/activate` |
| `NameError: name 'torch' is not defined` | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `connection refused` on DB | `docker-compose up -d` |
| `relation does not exist` | Re-run schema migration (see Database Setup) |
| `stocks table is empty` | Run `seed_stocks.py` |
| `0 articles found` in news ingestor | Check `NEWS_API_KEY` in `.env`. Use `--days 7`. |
| Feature rows show all NULL for a stock | Run `price_ingestor --mode symbol --symbol <TICKER>` then `feature_engine --mode symbol` |

---

## Roadmap

| Phase | Status | Deliverable |
|---|---|---|
| 1 | Complete | EDA scripts, charting, CSV pipeline |
| 2 | Complete | PostgreSQL pipeline, feature engine, FinBERT news NLP |
| 3 | Next | XGBoost model training, backtesting, prediction_engine service |
| 4 | Planned | FastAPI REST API, Next.js dashboard, Celery scheduler |