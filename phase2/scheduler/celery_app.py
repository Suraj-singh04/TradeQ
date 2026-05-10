"""
TradeQ — Scheduler
File: phase2/scheduler/celery_app.py

Celery application instance for TradeQ.
This is the entry point for both the worker and beat scheduler.

How to run:
  # Terminal 1 — executes tasks
  celery -A phase2.scheduler.celery_app worker --loglevel=info

  # Terminal 2 — triggers tasks on schedule
  celery -A phase2.scheduler.celery_app beat --loglevel=info

  # Inspect active tasks
  celery -A phase2.scheduler.celery_app inspect active

  # Force-run a specific task immediately (for testing)
  celery -A phase2.scheduler.celery_app call phase2.scheduler.tasks.run_price_ingestor

Author: TradeQ Project
"""

import sys
import logging
from pathlib import Path
from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv
import os

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

log = logging.getLogger(__name__)

# ─── Celery app ───────────────────────────────────────────────────────────────
app = Celery("tradeq")

app.config_from_object({
    # ── Broker and backend ────────────────────────────────────────────────────
    "broker_url":                os.getenv("CELERY_BROKER_URL",  "redis://localhost:6379/0"),
    "result_backend":            os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),

    # ── Serialization ─────────────────────────────────────────────────────────
    "task_serializer":           "json",
    "result_serializer":         "json",
    "accept_content":            ["json"],

    # ── Timezone — all schedules are in IST ───────────────────────────────────
    "timezone":                  "Asia/Kolkata",
    "enable_utc":                True,

    # ── Task behavior ─────────────────────────────────────────────────────────
    "task_acks_late":            True,    # Acknowledge only after task completes
    "task_reject_on_worker_lost":True,    # Re-queue if worker dies mid-task
    "task_track_started":        True,    # Track when tasks actually start
    "worker_prefetch_multiplier": 1,      # One task at a time per worker (heavy tasks)
    "task_soft_time_limit":      900,     # 15 min soft limit — logs warning
    "task_time_limit":           1200,    # 20 min hard limit — kills task

    # ── Result expiry ─────────────────────────────────────────────────────────
    "result_expires":            86400,   # Keep results for 24 hours

    # ── Beat schedule ─────────────────────────────────────────────────────────
    # All times are IST (Asia/Kolkata).
    # Market days: Monday–Friday.
    # Market hours: 09:15–15:30 IST.
    "beat_schedule": {

        # ── Pre-market data pipeline (runs before market opens) ───────────────
        "price-ingestor-daily": {
            "task":     "phase2.scheduler.tasks.run_price_ingestor",
            "schedule": crontab(hour=8, minute=0, day_of_week="mon-fri"),
            "options":  {"queue": "pipeline"},
        },
        "feature-engine-daily": {
            "task":     "phase2.scheduler.tasks.run_feature_engine",
            "schedule": crontab(hour=8, minute=30, day_of_week="mon-fri"),
            "options":  {"queue": "pipeline"},
        },
        "news-ingestor-morning": {
            "task":     "phase2.scheduler.tasks.run_news_ingestor",
            "schedule": crontab(hour=8, minute=45, day_of_week="mon-fri"),
            "options":  {"queue": "pipeline"},
        },

        # ── Prediction run (Phase 3 — task defined, activates after model trained)
        "prediction-engine-daily": {
            "task":     "phase2.scheduler.tasks.run_prediction_engine",
            "schedule": crontab(hour=9, minute=10, day_of_week="mon-fri"),
            "options":  {"queue": "ml"},
        },

        # ── Post-market updates ───────────────────────────────────────────────
        "news-ingestor-evening": {
            "task":     "phase2.scheduler.tasks.run_news_ingestor",
            "schedule": crontab(hour=15, minute=45, day_of_week="mon-fri"),
            "options":  {"queue": "pipeline"},
        },
        "feature-label-update": {
            "task":     "phase2.scheduler.tasks.run_label_update",
            "schedule": crontab(hour=16, minute=0, day_of_week="mon-fri"),
            "options":  {"queue": "pipeline"},
        },

        # ── Weekly model retraining (Phase 3) ────────────────────────────────
        "model-trainer-weekly": {
            "task":     "phase2.scheduler.tasks.run_model_trainer",
            "schedule": crontab(hour=2, minute=0, day_of_week="sun"),
            "options":  {"queue": "ml"},
        },
    },

    # ── Task routing — pipeline tasks and ML tasks on separate queues ─────────
    # This means a slow FinBERT run never blocks a price update
    "task_routes": {
        "phase2.scheduler.tasks.run_price_ingestor":   {"queue": "pipeline"},
        "phase2.scheduler.tasks.run_feature_engine":   {"queue": "pipeline"},
        "phase2.scheduler.tasks.run_news_ingestor":    {"queue": "pipeline"},
        "phase2.scheduler.tasks.run_label_update":     {"queue": "pipeline"},
        "phase2.scheduler.tasks.run_prediction_engine":{"queue": "ml"},
        "phase2.scheduler.tasks.run_model_trainer":    {"queue": "ml"},
    },
})

# Auto-discover tasks from this module
app.autodiscover_tasks(["phase2.scheduler"])