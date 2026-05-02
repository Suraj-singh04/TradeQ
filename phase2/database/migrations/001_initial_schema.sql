-- ============================================================
-- TradeQ — Database Schema
-- File: phase2/database/migrations/001_initial_schema.sql
--
-- This file runs automatically on first `docker-compose up`
-- because it lives in /docker-entrypoint-initdb.d/
--
-- To run manually against a live database:
--   docker exec -it tradeq_postgres psql -U tradeq -d tradeq_db \
--     -f /docker-entrypoint-initdb.d/001_initial_schema.sql
--
-- Tables created:
--   1. stocks          — master list of all tracked companies
--   2. ohlcv_daily     — price/volume history (TimescaleDB hypertable)
--   3. features_daily  — all 50 engineered features per stock per day
--   4. news_sentiment  — FinBERT sentiment scores per stock per day
--   5. predictions     — model output, confidence scores, actual outcomes
-- ============================================================


-- ─── Enable TimescaleDB extension ────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;


-- ─── 1. stocks ───────────────────────────────────────────────────────────────
-- Master reference table. Every other table references this via stock_id.
-- Populated once by the seed script — rarely changes.

CREATE TABLE IF NOT EXISTS stocks (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20)  NOT NULL UNIQUE,   -- e.g. "RELIANCE.NS"
    name            VARCHAR(200) NOT NULL,           -- e.g. "Reliance Industries Ltd"
    sector          VARCHAR(100),                    -- e.g. "energy"
    industry        VARCHAR(100),                    -- e.g. "Oil & Gas"
    market_cap_cr   BIGINT,                          -- Market cap in INR crores
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    added_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Index for fast symbol lookups (used in every query)
CREATE INDEX IF NOT EXISTS idx_stocks_symbol    ON stocks (symbol);
CREATE INDEX IF NOT EXISTS idx_stocks_sector    ON stocks (sector);
CREATE INDEX IF NOT EXISTS idx_stocks_is_active ON stocks (is_active);

COMMENT ON TABLE stocks IS 'Master list of all NSE stocks tracked by TradeQ';
COMMENT ON COLUMN stocks.symbol IS 'Yahoo Finance ticker with .NS suffix';
COMMENT ON COLUMN stocks.market_cap_cr IS 'Market capitalisation in Indian Rupee crores';


-- ─── 2. ohlcv_daily ──────────────────────────────────────────────────────────
-- Every daily price candle for every stock.
-- This is a TimescaleDB hypertable — partitioned by time automatically.
-- Queries like "last 30 days for RELIANCE" are 100x faster than plain Postgres.

CREATE TABLE IF NOT EXISTS ohlcv_daily (
    time            TIMESTAMPTZ  NOT NULL,           -- Trading date (always midnight IST)
    stock_id        INT          NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    open            NUMERIC(12,2) NOT NULL,
    high            NUMERIC(12,2) NOT NULL,
    low             NUMERIC(12,2) NOT NULL,
    close           NUMERIC(12,2) NOT NULL,
    volume          BIGINT        NOT NULL,
    adjusted_close  NUMERIC(12,2),                  -- Split/dividend adjusted close
    source          VARCHAR(20)  NOT NULL DEFAULT 'yahoo', -- yahoo / nse / kite
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Prevent duplicate rows for same stock on same day
    CONSTRAINT uq_ohlcv_stock_date UNIQUE (time, stock_id)
);

-- Convert to TimescaleDB hypertable — partitioned by month
-- This is what makes time-range queries fast at scale
SELECT create_hypertable(
    'ohlcv_daily', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Indexes for the most common query patterns
CREATE INDEX IF NOT EXISTS idx_ohlcv_stock_id  ON ohlcv_daily (stock_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_time      ON ohlcv_daily (time DESC);

COMMENT ON TABLE ohlcv_daily IS 'Daily OHLCV price data — TimescaleDB hypertable partitioned by month';
COMMENT ON COLUMN ohlcv_daily.adjusted_close IS 'Close price adjusted for splits and dividends — use this for ML features';


-- ─── 3. features_daily ───────────────────────────────────────────────────────
-- All 50 engineered features per stock per day.
-- Written by feature_engine service after ohlcv_daily is populated.
-- Read by the ML model training pipeline in Phase 3.
-- Each column name matches exactly the feature list in config.py.

CREATE TABLE IF NOT EXISTS features_daily (
    time            TIMESTAMPTZ  NOT NULL,
    stock_id        INT          NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,

    -- ── Technical indicators ─────────────────────────────────────────────────
    rsi_14          NUMERIC(7,3),   -- RSI (0-100)
    macd            NUMERIC(10,4),  -- MACD line value
    macd_signal     NUMERIC(10,4),  -- MACD signal line
    macd_hist       NUMERIC(10,4),  -- MACD histogram (macd - signal)
    bb_upper        NUMERIC(12,2),  -- Bollinger Band upper
    bb_lower        NUMERIC(12,2),  -- Bollinger Band lower
    bb_pct_b        NUMERIC(7,4),   -- %B — where price sits within bands (0-1)
    bb_width        NUMERIC(7,4),   -- Band width — volatility measure
    ema_20          NUMERIC(12,2),  -- 20-day EMA
    ema_50          NUMERIC(12,2),  -- 50-day EMA
    ema_200         NUMERIC(12,2),  -- 200-day EMA (long-term trend)
    ema_cross_20_50 NUMERIC(6,4),   -- EMA20/EMA50 ratio (> 1 = bullish)
    atr_14          NUMERIC(10,4),  -- Average True Range
    adx_14          NUMERIC(7,3),   -- ADX — trend strength (not direction)
    obv             BIGINT,         -- On-Balance Volume (cumulative)
    vwap_dev        NUMERIC(7,4),   -- % deviation from VWAP
    stoch_k         NUMERIC(7,3),   -- Stochastic %K
    stoch_d         NUMERIC(7,3),   -- Stochastic %D
    cci_20          NUMERIC(9,3),   -- Commodity Channel Index
    williams_r      NUMERIC(7,3),   -- Williams %R

    -- ── Price action features ─────────────────────────────────────────────────
    daily_return    NUMERIC(8,4),   -- % daily return
    log_return      NUMERIC(8,4),   -- Log return
    body_pct        NUMERIC(7,4),   -- Candle body size as % of price
    gap_pct         NUMERIC(7,4),   -- Gap from prev close to today open (%)
    high_52w_dist   NUMERIC(7,4),   -- % distance from 52-week high
    low_52w_dist    NUMERIC(7,4),   -- % distance from 52-week low
    close_position  NUMERIC(6,4),   -- Where close is in today's range (0-1)
    consec_green    SMALLINT,       -- Consecutive green candles
    consec_red      SMALLINT,       -- Consecutive red candles
    volatility_20d  NUMERIC(7,4),   -- Rolling 20-day annualised volatility

    -- ── Volume features ────────────────────────────────────────────────────────
    volume_spike    NUMERIC(7,3),   -- Today's volume / 20-day avg volume
    volume_ma_20    BIGINT,         -- 20-day volume moving average
    obv_slope       NUMERIC(10,4),  -- OBV 5-day slope (buying/selling pressure trend)

    -- ── Sentiment & news features (written by news_ingestor) ─────────────────
    news_sentiment  NUMERIC(5,3),   -- FinBERT composite score (-1 to +1)
    news_count      SMALLINT,       -- Articles in last 24h
    news_positive   SMALLINT,       -- Count of positive articles
    news_negative   SMALLINT,       -- Count of negative articles

    -- ── Fundamental / event features ─────────────────────────────────────────
    days_to_earnings    SMALLINT,   -- Trading days until next earnings report
    earnings_surprise   NUMERIC(7,3), -- Last EPS surprise % (positive = beat)
    analyst_upside      NUMERIC(7,3), -- % upside to consensus analyst target

    -- ── Macro & market context features ─────────────────────────────────────
    nifty_daily_ret     NUMERIC(8,4), -- Nifty 50 daily return
    banknifty_daily_ret NUMERIC(8,4), -- Bank Nifty daily return
    india_vix           NUMERIC(7,3), -- India VIX level
    sector_daily_ret    NUMERIC(8,4), -- Stock's sector index daily return
    relative_strength   NUMERIC(8,4), -- Stock return - Nifty return
    usd_inr_change      NUMERIC(7,4), -- USD/INR daily % change
    crude_change        NUMERIC(7,4), -- Crude oil daily % change
    fii_net_cr          NUMERIC(12,2),-- FII net buy/sell in INR crores
    dii_net_cr          NUMERIC(12,2),-- DII net buy/sell in INR crores
    market_breadth      NUMERIC(6,3), -- Advance/decline ratio for NSE

    -- ── ML target label (filled by feature_engine at EOD) ───────────────────
    -- Did this stock beat Nifty by > 1% the NEXT day?
    -- 1 = yes (buy signal), 0 = no, NULL = not yet known (today's data)
    label               SMALLINT,
    next_day_return     NUMERIC(8,4), -- Actual next-day return (filled EOD)

    -- ── Metadata ──────────────────────────────────────────────────────────────
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_features_stock_date UNIQUE (time, stock_id)
);

-- Convert to hypertable
SELECT create_hypertable(
    'features_daily', 'time',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_features_stock_time ON features_daily (stock_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_features_time       ON features_daily (time DESC);
CREATE INDEX IF NOT EXISTS idx_features_label      ON features_daily (label) WHERE label IS NOT NULL;

COMMENT ON TABLE features_daily IS 'All 50 engineered ML features per stock per day — written by feature_engine';
COMMENT ON COLUMN features_daily.label IS '1 = stock beat Nifty by >1% next day, 0 = did not. NULL = future (unknown).';


-- ─── 4. news_sentiment ───────────────────────────────────────────────────────
-- Raw news records with FinBERT sentiment scores.
-- Kept separate from features_daily so we can trace every score back to
-- its source article. The aggregated scores are copied to features_daily.

CREATE TABLE IF NOT EXISTS news_sentiment (
    id              BIGSERIAL    PRIMARY KEY,
    stock_id        INT          NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    published_at    TIMESTAMPTZ  NOT NULL,           -- When the article was published
    fetched_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    headline        TEXT         NOT NULL,
    source          VARCHAR(100),                    -- e.g. "Economic Times"
    url             TEXT,
    sentiment_label VARCHAR(10)  NOT NULL,           -- "positive" / "negative" / "neutral"
    sentiment_score NUMERIC(5,3) NOT NULL,           -- FinBERT confidence (-1 to +1)
    is_processed    BOOLEAN      NOT NULL DEFAULT FALSE -- Has this been rolled into features?
);

CREATE INDEX IF NOT EXISTS idx_news_stock_date ON news_sentiment (stock_id, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_date       ON news_sentiment (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_processed  ON news_sentiment (is_processed) WHERE NOT is_processed;

COMMENT ON TABLE news_sentiment IS 'Raw news articles with FinBERT sentiment scores — aggregated into features_daily daily';


-- ─── 5. predictions ──────────────────────────────────────────────────────────
-- Model output for each trading day.
-- Written by prediction_engine in Phase 3.
-- actual_return is filled at EOD so we can track the model's real win rate.

CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGSERIAL    PRIMARY KEY,
    prediction_date     DATE         NOT NULL,       -- Which trading day this is for
    stock_id            INT          NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    rank                SMALLINT,                    -- 1 = top pick that day
    confidence_score    NUMERIC(5,3) NOT NULL,       -- Model probability (0.0 - 1.0)
    model_version       VARCHAR(30)  NOT NULL,       -- e.g. "tradeq_v1.2"
    top_factors         JSONB,                       -- Top 5 features driving prediction
    -- Filled at EOD to measure accuracy
    actual_return       NUMERIC(8,4),                -- Stock's actual return that day
    beat_nifty          BOOLEAN,                     -- Did it beat Nifty by > 1%?
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_prediction_stock_date UNIQUE (prediction_date, stock_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_date       ON predictions (prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_pred_stock      ON predictions (stock_id, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_pred_rank       ON predictions (prediction_date, rank);
CREATE INDEX IF NOT EXISTS idx_pred_confidence ON predictions (confidence_score DESC);

COMMENT ON TABLE predictions IS 'Daily ML model predictions — rank + confidence per stock. actual_return filled EOD.';
COMMENT ON COLUMN predictions.top_factors IS 'JSONB array of top 5 feature names and their SHAP values';


-- ─── Convenience view: latest features per stock ──────────────────────────────
-- Used by the API gateway and dashboard to get current snapshot fast.
CREATE OR REPLACE VIEW v_latest_features AS
SELECT DISTINCT ON (stock_id)
    f.*,
    s.symbol,
    s.name,
    s.sector
FROM features_daily f
JOIN stocks s ON s.id = f.stock_id
ORDER BY stock_id, time DESC;

COMMENT ON VIEW v_latest_features IS 'Most recent feature row per stock — used by API for live deep-dive data';


-- ─── Convenience view: today predictions with stock info ─────────────────────
CREATE OR REPLACE VIEW v_today_predictions AS
SELECT
    p.rank,
    p.confidence_score,
    p.prediction_date,
    p.top_factors,
    s.symbol,
    s.name,
    s.sector,
    f.rsi_14,
    f.macd_hist,
    f.volume_spike,
    f.news_sentiment,
    f.india_vix,
    f.relative_strength
FROM predictions p
JOIN stocks s ON s.id = p.stock_id
LEFT JOIN v_latest_features f ON f.stock_id = p.stock_id
WHERE p.prediction_date = CURRENT_DATE
ORDER BY p.rank;

COMMENT ON VIEW v_today_predictions IS 'Todays ranked predictions joined with key features — served by /predictions/today API';


-- ─── Schema version tracking ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_versions (
    version     VARCHAR(20) PRIMARY KEY,
    description TEXT,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO schema_versions (version, description)
VALUES ('001', 'Initial schema — stocks, ohlcv_daily, features_daily, news_sentiment, predictions')
ON CONFLICT (version) DO NOTHING;


-- ─── Done ─────────────────────────────────────────────────────────────────────
DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'TradeQ schema created successfully.';
    RAISE NOTICE 'Tables: stocks, ohlcv_daily, features_daily, news_sentiment, predictions';
    RAISE NOTICE 'Views:  v_latest_features, v_today_predictions';
    RAISE NOTICE 'TimescaleDB hypertables: ohlcv_daily, features_daily';
    RAISE NOTICE '============================================================';
END $$;