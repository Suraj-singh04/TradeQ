"""
TradeQ — Scheduler
File: phase2/scheduler/run_scheduler.py

Management script for the TradeQ Celery scheduler.
Provides a single entry point to start, stop, and monitor
the worker and beat processes.

Usage:
  # Start both worker and beat (recommended for development)
  python3 phase2/scheduler/run_scheduler.py start

  # Start worker only
  python3 phase2/scheduler/run_scheduler.py worker

  # Start beat only (requires worker already running)
  python3 phase2/scheduler/run_scheduler.py beat

  # Show scheduled tasks and next run times
  python3 phase2/scheduler/run_scheduler.py status

  # Manually trigger a specific task right now (for testing)
  python3 phase2/scheduler/run_scheduler.py trigger --task price
  python3 phase2/scheduler/run_scheduler.py trigger --task features
  python3 phase2/scheduler/run_scheduler.py trigger --task news
  python3 phase2/scheduler/run_scheduler.py trigger --task labels

Author: TradeQ Project
"""

import sys
import subprocess
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CELERY_APP = "phase2.scheduler.celery_app"

TASK_MAP = {
    "price":    "phase2.scheduler.tasks.run_price_ingestor",
    "features": "phase2.scheduler.tasks.run_feature_engine",
    "news":     "phase2.scheduler.tasks.run_news_ingestor",
    "labels":   "phase2.scheduler.tasks.run_label_update",
    "predict":  "phase2.scheduler.tasks.run_prediction_engine",
    "train":    "phase2.scheduler.tasks.run_model_trainer",
}


def check_redis() -> bool:
    """Verify Redis is reachable before starting Celery."""
    import os
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        r.ping()
        log.info("Redis connection: OK")
        return True
    except Exception as e:
        log.error(f"Redis not reachable: {e}")
        log.error("Make sure Docker is running: docker-compose up -d")
        return False


def check_db() -> bool:
    """Verify PostgreSQL is reachable before starting."""
    import os
    try:
        import sqlalchemy as sa
        from sqlalchemy import text
        engine = sa.create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("PostgreSQL connection: OK")
        return True
    except Exception as e:
        log.error(f"PostgreSQL not reachable: {e}")
        log.error("Make sure Docker is running: docker-compose up -d")
        return False


def start_both():
    """
    Start worker and beat as separate processes using Popen.
    Both run in the foreground — Ctrl+C stops both.
    """
    if not check_redis() or not check_db():
        sys.exit(1)

    log.info("Starting TradeQ Celery worker + beat scheduler...")
    log.info("Press Ctrl+C to stop both processes.")
    log.info("─" * 55)

    # Log file paths
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    worker_cmd = [
        "celery", "-A", CELERY_APP, "worker",
        "--loglevel=info",
        "--queues=pipeline,ml",
        "--concurrency=2",          # 2 concurrent tasks max (low-spec machine)
        "--logfile", str(log_dir / "celery_worker.log"),
        "--pidfile", str(log_dir / "celery_worker.pid"),
    ]
    beat_cmd = [
        "celery", "-A", CELERY_APP, "beat",
        "--loglevel=info",
        "--logfile", str(log_dir / "celery_beat.log"),
        "--pidfile",  str(log_dir / "celery_beat.pid"),
        "--schedule", str(log_dir / "celerybeat-schedule"),
    ]

    log.info(f"Worker log: {log_dir}/celery_worker.log")
    log.info(f"Beat log:   {log_dir}/celery_beat.log")
    log.info("─" * 55)

    try:
        worker_proc = subprocess.Popen(worker_cmd, cwd=str(ROOT))
        time.sleep(3)   # Give worker time to start before beat connects
        beat_proc   = subprocess.Popen(beat_cmd, cwd=str(ROOT))

        log.info(f"Worker PID: {worker_proc.pid}")
        log.info(f"Beat PID:   {beat_proc.pid}")
        log.info("Scheduler is running. Watching logs...")

        # Tail worker log to console
        worker_proc.wait()
        beat_proc.wait()

    except KeyboardInterrupt:
        log.info("Shutting down...")
        worker_proc.terminate()
        beat_proc.terminate()
        worker_proc.wait()
        beat_proc.wait()
        log.info("Both processes stopped.")


def start_worker():
    """Start worker process only."""
    if not check_redis() or not check_db():
        sys.exit(1)

    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting Celery worker (queues: pipeline, ml)...")
    cmd = [
        "celery", "-A", CELERY_APP, "worker",
        "--loglevel=info",
        "--queues=pipeline,ml",
        "--concurrency=2",
    ]
    subprocess.run(cmd, cwd=str(ROOT))


def start_beat():
    """Start beat scheduler only."""
    if not check_redis():
        sys.exit(1)

    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting Celery beat scheduler...")
    cmd = [
        "celery", "-A", CELERY_APP, "beat",
        "--loglevel=info",
        "--schedule", str(log_dir / "celerybeat-schedule"),
    ]
    subprocess.run(cmd, cwd=str(ROOT))


def show_status():
    """Show schedule and next run times for all tasks."""
    from phase2.scheduler.celery_app import app
    from celery.schedules import crontab
    from zoneinfo import ZoneInfo
    from datetime import datetime

    ist = ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)

    print("\n" + "═" * 65)
    print(f"  TradeQ Scheduler Status — {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("═" * 65)
    print(f"  Broker : {app.conf.broker_url}")
    print(f"  TZ     : {app.conf.timezone}")
    print()
    print(f"  {'Task':<35} {'Schedule (IST)':<20} {'Queue'}")
    print(f"  {'─'*35} {'─'*20} {'─'*10}")

    schedule_display = {
        "price-ingestor-daily":    ("Mon-Fri 08:00", "pipeline"),
        "feature-engine-daily":    ("Mon-Fri 08:30", "pipeline"),
        "news-ingestor-morning":   ("Mon-Fri 08:45", "pipeline"),
        "prediction-engine-daily": ("Mon-Fri 09:10", "ml      [Phase 3]"),
        "news-ingestor-evening":   ("Mon-Fri 15:45", "pipeline"),
        "feature-label-update":    ("Mon-Fri 16:00", "pipeline"),
        "model-trainer-weekly":    ("Sunday  02:00", "ml      [Phase 3]"),
    }

    for name, (schedule, queue) in schedule_display.items():
        print(f"  {name:<35} {schedule:<20} {queue}")

    print()
    print("  Trigger a task manually:")
    for short, full in TASK_MAP.items():
        print(f"    python3 phase2/scheduler/run_scheduler.py trigger --task {short}")

    print("═" * 65 + "\n")

    # Check Redis connectivity
    check_redis()
    check_db()


def trigger_task(task_short: str):
    """Manually trigger a task immediately — useful for testing."""
    if task_short not in TASK_MAP:
        log.error(f"Unknown task '{task_short}'. Valid options: {list(TASK_MAP.keys())}")
        sys.exit(1)

    if not check_redis():
        sys.exit(1)

    task_name = TASK_MAP[task_short]
    log.info(f"Triggering task: {task_name}")
    log.info("This sends the task to the queue — make sure a worker is running.")
    log.info("Start a worker with: python3 phase2/scheduler/run_scheduler.py worker")

    from phase2.scheduler.celery_app import app
    result = app.send_task(task_name)

    log.info(f"Task queued — ID: {result.id}")
    log.info("Waiting for result (timeout: 600s)...")

    try:
        output = result.get(timeout=600)
        log.info(f"Task completed: {output}")
    except Exception as e:
        log.error(f"Task failed or timed out: {e}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TradeQ scheduler management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  start    Start both worker and beat (recommended)
  worker   Start worker only
  beat     Start beat only
  status   Show schedule and connectivity status
  trigger  Manually run a task now (requires worker running)

Trigger task names:
  price     Daily OHLCV update
  features  Daily feature computation
  news      News fetch + FinBERT scoring
  labels    Post-market label fill
  predict   Prediction engine [Phase 3]
  train     Weekly model retraining [Phase 3]
        """
    )
    parser.add_argument("command", choices=["start", "worker", "beat", "status", "trigger"])
    parser.add_argument("--task", type=str, default=None,
                        help="Task name for trigger command")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "start":
        start_both()
    elif args.command == "worker":
        start_worker()
    elif args.command == "beat":
        start_beat()
    elif args.command == "status":
        show_status()
    elif args.command == "trigger":
        if not args.task:
            log.error("--task required for trigger command")
            sys.exit(1)
        trigger_task(args.task)


if __name__ == "__main__":
    main()