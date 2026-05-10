"""
TradeQ — Scheduler
File: phase2/scheduler/tasks.py

Celery task definitions for all TradeQ pipeline services.

Key behavior:
  - worker_ready signal: fires on every worker start, checks if today's
    data is missing, and triggers the full pipeline immediately if it is.
    This means no matter what time you turn your PC on, TradeQ catches up.
  - All tasks are idempotent — safe to re-run if they fail and retry.
  - Tasks chain in correct order: price → features → news.
  - Phase 3 tasks (prediction, training) are defined but skip gracefully
    until the model is built.

Author: TradeQ Project
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase2.scheduler.celery_app import app
from celery import chain
from celery.signals import worker_ready
from celery.utils.log import get_task_logger
from dotenv import load_dotenv
import os

load_dotenv(ROOT / ".env")

log = get_task_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_engine():
    import sqlalchemy as sa
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL not set in .env")
    return sa.create_engine(db_url, pool_pre_ping=True, pool_size=5)


def _is_weekday() -> bool:
    return datetime.now(IST).weekday() < 5


def _get_latest_ohlcv_date(engine) -> date | None:
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(time)::date FROM ohlcv_daily")
        ).scalar()
    return result


def _get_latest_feature_date(engine) -> date | None:
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(time)::date FROM features_daily")
        ).scalar()
    return result


# ─── Startup trigger ──────────────────────────────────────────────────────────

@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """
    Fires automatically every time the Celery worker starts.

    Scenarios handled:
      PC on at 08:00 → beat fires tasks at scheduled times (no catch-up needed)
      PC on at 12:00 → startup detects missing data, triggers pipeline immediately
      PC on at 22:00 → same as above, catches up for that day
      PC on weekend  → skips entirely (correct behavior)
      PC was off 3 days → catches up all missing days via daily mode
    """
    log.info("=" * 55)
    log.info("[startup] TradeQ worker started — checking data status")
    log.info(f"[startup] Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
    log.info("=" * 55)

    if not _is_weekday():
        log.info("[startup] Weekend — pipeline check skipped")
        return

    try:
        engine      = _get_engine()
        today       = date.today()
        latest_ohlcv= _get_latest_ohlcv_date(engine)
        latest_feat = _get_latest_feature_date(engine)

        log.info(f"[startup] Today          : {today}")
        log.info(f"[startup] Latest OHLCV   : {latest_ohlcv}")
        log.info(f"[startup] Latest features: {latest_feat}")

        # Case 1: OHLCV is behind — run full pipeline
        if latest_ohlcv is None or latest_ohlcv < today:
            days_behind = (today - latest_ohlcv).days if latest_ohlcv else "unknown"
            log.info(f"[startup] OHLCV is {days_behind} day(s) behind — running full pipeline now")
            chain(
                run_price_ingestor.si(),
                run_feature_engine.si(),
                run_news_ingestor.si(),
            ).delay()
            return

        # Case 2: OHLCV current but features behind — run feature + news only
        if latest_feat is None or latest_feat < today:
            log.info("[startup] OHLCV current but features behind — running feature + news pipeline")
            chain(
                run_feature_engine.si(),
                run_news_ingestor.si(),
            ).delay()
            return

        # Case 3: Everything is current
        log.info("[startup] All data is current — no catch-up needed")

    except Exception as e:
        log.error(f"[startup] Pipeline check failed: {e}", exc_info=True)
        log.error("[startup] Scheduled tasks will still run at their defined times")


# ─── Task 1: Price Ingestor ───────────────────────────────────────────────────

@app.task(
    name="phase2.scheduler.tasks.run_price_ingestor",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    soft_time_limit=600,
    time_limit=900,
)
def run_price_ingestor(self):
    """
    Daily OHLCV update for all active stocks.
    Fetches last 5 days and inserts missing rows.
    Scheduled: 08:00 IST Mon-Fri.
    Also triggered on startup if data is behind.
    """
    task_start = time.time()
    log.info(f"[price_ingestor] Starting — {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")

    try:
        from phase2.services.price_ingestor.ingestor import (
            get_engine, get_stock_map, run_daily, verify_database
        )
        engine    = get_engine()
        stock_map = get_stock_map(engine)
        run_daily(engine, stock_map)
        verify_database(engine)

        elapsed = round(time.time() - task_start, 1)
        log.info(f"[price_ingestor] Completed in {elapsed}s")
        return {"status": "ok", "elapsed_seconds": elapsed}

    except Exception as exc:
        elapsed = round(time.time() - task_start, 1)
        log.error(f"[price_ingestor] Failed after {elapsed}s: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ─── Task 2: Feature Engine ───────────────────────────────────────────────────

@app.task(
    name="phase2.scheduler.tasks.run_feature_engine",
    bind=True,
    max_retries=2,
    default_retry_delay=600,
    soft_time_limit=900,
    time_limit=1200,
)
def run_feature_engine(self):
    """
    Daily feature computation for all active stocks.
    Reads OHLCV, computes 30+ indicators, downloads macro data.
    Scheduled: 08:30 IST Mon-Fri.
    Also runs after price_ingestor in startup catch-up chain.
    """
    task_start = time.time()
    log.info(f"[feature_engine] Starting — {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")

    try:
        from phase2.services.feature_engine.engine import (
            get_engine, get_stock_map, load_macro_data, run_daily, run_verify
        )
        engine    = get_engine()
        stock_map = get_stock_map(engine)
        log.info("[feature_engine] Loading macro data...")
        macro = load_macro_data(engine)
        log.info(f"[feature_engine] Macro: {len(macro)} rows, cols: {list(macro.columns)}")
        run_daily(engine, stock_map, macro)
        run_verify(engine)

        elapsed = round(time.time() - task_start, 1)
        log.info(f"[feature_engine] Completed in {elapsed}s")
        return {"status": "ok", "elapsed_seconds": elapsed}

    except Exception as exc:
        elapsed = round(time.time() - task_start, 1)
        log.error(f"[feature_engine] Failed after {elapsed}s: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ─── Task 3: News Ingestor ────────────────────────────────────────────────────

@app.task(
    name="phase2.scheduler.tasks.run_news_ingestor",
    bind=True,
    max_retries=2,
    default_retry_delay=300,
    soft_time_limit=600,
    time_limit=900,
)
def run_news_ingestor(self):
    """
    Fetch today's headlines, score with FinBERT, aggregate into features_daily.
    Scheduled: 08:45 IST and 15:45 IST Mon-Fri.
    Also runs at end of startup catch-up chain.
    """
    task_start = time.time()
    log.info(f"[news_ingestor] Starting — {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")

    try:
        from phase2.services.news_ingestor.ingestor import (
            get_engine, get_stock_map, run_fetch, run_verify
        )
        engine    = get_engine()
        stock_map = get_stock_map(engine)
        run_fetch(engine, stock_map, days_back=1)
        run_verify(engine)

        elapsed = round(time.time() - task_start, 1)
        log.info(f"[news_ingestor] Completed in {elapsed}s")
        return {"status": "ok", "elapsed_seconds": elapsed}

    except Exception as exc:
        elapsed = round(time.time() - task_start, 1)
        log.error(f"[news_ingestor] Failed after {elapsed}s: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ─── Task 4: Label Update ─────────────────────────────────────────────────────

@app.task(
    name="phase2.scheduler.tasks.run_label_update",
    bind=True,
    max_retries=2,
    default_retry_delay=300,
    soft_time_limit=600,
    time_limit=900,
)
def run_label_update(self):
    """
    Post-market label fill — recomputes labels for last 3 days.
    Scheduled: 16:00 IST Mon-Fri.
    """
    task_start = time.time()
    log.info(f"[label_update] Starting — {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")

    try:
        from phase2.services.feature_engine.engine import (
            get_engine, get_stock_map, load_macro_data, run_daily
        )
        from sqlalchemy import text

        engine    = get_engine()
        stock_map = get_stock_map(engine)
        macro     = load_macro_data(engine)
        run_daily(engine, stock_map, macro)

        with engine.connect() as conn:
            null_count = conn.execute(text("""
                SELECT COUNT(*) FROM features_daily
                WHERE label IS NULL
                  AND time < NOW() - INTERVAL '1 day'
            """)).scalar()

        elapsed = round(time.time() - task_start, 1)
        log.info(f"[label_update] {null_count} NULL labels remain (excl. today) — done in {elapsed}s")
        return {"status": "ok", "null_labels_remaining": null_count, "elapsed_seconds": elapsed}

    except Exception as exc:
        elapsed = round(time.time() - task_start, 1)
        log.error(f"[label_update] Failed after {elapsed}s: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ─── Task 5: Prediction Engine (Phase 3 placeholder) ─────────────────────────

@app.task(
    name="phase2.scheduler.tasks.run_prediction_engine",
    bind=True,
    max_retries=1,
    default_retry_delay=120,
    soft_time_limit=300,
    time_limit=600,
)
def run_prediction_engine(self):
    """
    Scores all stocks with trained XGBoost model, writes to predictions table.
    Scheduled: 09:10 IST Mon-Fri (before market opens at 09:15).
    Phase 3: PLACEHOLDER — activates after model is trained.
    """
    log.info("[prediction_engine] Phase 3 not yet built — skipping")
    return {"status": "skipped", "reason": "Phase 3 not yet implemented"}


# ─── Task 6: Model Trainer (Phase 3 placeholder) ─────────────────────────────

@app.task(
    name="phase2.scheduler.tasks.run_model_trainer",
    bind=True,
    max_retries=1,
    default_retry_delay=600,
    soft_time_limit=3600,
    time_limit=4800,
)
def run_model_trainer(self):
    """
    Weekly XGBoost retraining on latest features_daily data.
    Scheduled: Sunday 02:00 IST.
    Phase 3: PLACEHOLDER — activates after Phase 3 is built.
    """
    log.info("[model_trainer] Phase 3 not yet built — skipping")
    return {"status": "skipped", "reason": "Phase 3 not yet implemented"}