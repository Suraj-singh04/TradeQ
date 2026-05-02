"""
TradeQ — Phase 2
File: phase2/services/price_ingestor/ingestor.py

What this service does:
  - Reads all active stocks from the stocks table
  - Downloads OHLCV data from yfinance for each stock
  - Writes directly into the ohlcv_daily TimescaleDB hypertable
  - Handles three modes:
      * backfill  — downloads full history (run once to populate from scratch)
      * daily     — downloads only missing days since last record (run every day)
      * symbol    — downloads one specific stock (for debugging or recovery)
  - Migrates your existing data/raw/ CSVs into the database automatically
  - Never creates duplicate rows — uses INSERT ON CONFLICT DO NOTHING
  - Logs every action with timestamps

How to run:
  # Fill the entire database from scratch (run once)
  python3 phase2/services/price_ingestor/ingestor.py --mode backfill

  # Daily update — only fetch what's missing (run every morning)
  python3 phase2/services/price_ingestor/ingestor.py --mode daily

  # Single stock recovery
  python3 phase2/services/price_ingestor/ingestor.py --mode symbol --symbol RELIANCE.NS

  # Migrate existing CSVs from data/raw/ into the database
  python3 phase2/services/price_ingestor/ingestor.py --mode migrate

Author: TradeQ Project
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import CONFIG, get_symbols, get_raw_path
import yfinance as yf
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv
import os

load_dotenv(ROOT / ".env")

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s  %(levelname)s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
IST          = ZoneInfo(CONFIG["project"]["timezone"])
PAUSE        = CONFIG["download"]["pause_seconds"]
MAX_RETRIES  = CONFIG["download"]["max_retries"]
BACKFILL_PERIOD = CONFIG["download"]["training_period"]   # "5y"
DAILY_PERIOD    = "5d"    # Fetch last 5 days and let ON CONFLICT handle deduplication


# ─── Database ─────────────────────────────────────────────────────────────────

def get_engine() -> sa.Engine:
    """Create SQLAlchemy engine. Raises clearly if .env is missing."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL not set in .env")
    return sa.create_engine(db_url, pool_pre_ping=True, pool_size=5)


def get_stock_map(engine: sa.Engine) -> dict[str, int]:
    """
    Return a mapping of symbol → stock_id for all active stocks.
    Used to look up the integer FK when inserting OHLCV rows.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT symbol, id FROM stocks WHERE is_active = TRUE")
        ).fetchall()
    if not rows:
        raise RuntimeError(
            "stocks table is empty. Run seed_stocks.py first."
        )
    return {row.symbol: row.id for row in rows}


def get_last_date(engine: sa.Engine, stock_id: int) -> date | None:
    """
    Return the most recent date we have data for a given stock_id.
    Returns None if no data exists yet.
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(time) FROM ohlcv_daily WHERE stock_id = :sid"),
            {"sid": stock_id}
        ).scalar()
    return result.date() if result else None


# ─── Download ─────────────────────────────────────────────────────────────────

def download_ohlcv(symbol: str, period: str, retries: int = MAX_RETRIES) -> pd.DataFrame | None:
    """
    Download OHLCV data from yfinance with retry logic.

    Args:
        symbol:  NSE ticker e.g. "RELIANCE.NS"
        period:  yfinance period string e.g. "5y", "1y", "5d"
        retries: How many times to retry on failure

    Returns:
        Cleaned DataFrame or None if all retries failed.
    """
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            df     = ticker.history(period=period, auto_adjust=True)

            if df.empty:
                log.warning(f"  {symbol} — yfinance returned empty DataFrame")
                return None

            # Clean up
            df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            df = df[["Open", "High", "Low", "Close", "Volume"]].round(2)
            df = df.dropna(subset=["Close"])
            df = df[df["Volume"] > 0]    # Drop non-trading days
            return df

        except Exception as e:
            log.warning(f"  {symbol} — attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s, 8s

    log.error(f"  {symbol} — all {retries} attempts failed")
    return None


# ─── Insert ───────────────────────────────────────────────────────────────────

INSERT_SQL = text("""
    INSERT INTO ohlcv_daily
        (time, stock_id, open, high, low, close, volume, adjusted_close, source)
    VALUES
        (:time, :stock_id, :open, :high, :low, :close, :volume, :close, 'yahoo')
    ON CONFLICT (time, stock_id) DO NOTHING
""")


def insert_ohlcv(engine: sa.Engine, stock_id: int, df: pd.DataFrame) -> int:
    """
    Insert a DataFrame of OHLCV rows into ohlcv_daily.
    ON CONFLICT DO NOTHING means re-running is always safe.

    Returns:
        Number of rows actually inserted.
    """
    rows = []
    for dt, row in df.iterrows():
        rows.append({
            "time":     datetime.combine(dt.date(), datetime.min.time()),
            "stock_id": stock_id,
            "open":     float(row["Open"]),
            "high":     float(row["High"]),
            "low":      float(row["Low"]),
            "close":    float(row["Close"]),
            "volume":   int(row["Volume"]),
        })

    if not rows:
        return 0

    with engine.begin() as conn:
        result = conn.execute(INSERT_SQL, rows)

    return result.rowcount


# ─── Modes ────────────────────────────────────────────────────────────────────

def run_backfill(engine: sa.Engine, stock_map: dict[str, int]) -> None:
    """
    Download full history (BACKFILL_PERIOD = 5 years) for all active stocks.
    Run once to populate the database from scratch.
    """
    log.info(f"MODE: backfill — downloading {BACKFILL_PERIOD} of history for {len(stock_map)} stocks")
    log.info("This will take several minutes. Grab a chai.")
    log.info("─" * 55)

    total_rows = 0
    success    = 0
    failed     = []

    for i, (symbol, stock_id) in enumerate(stock_map.items(), 1):
        log.info(f"[{i:>2}/{len(stock_map)}] {symbol}")

        df = download_ohlcv(symbol, BACKFILL_PERIOD)
        if df is None:
            failed.append(symbol)
            continue

        inserted  = insert_ohlcv(engine, stock_id, df)
        total_rows += inserted
        success    += 1
        log.info(f"  {symbol:<20} {len(df)} rows downloaded  {inserted} inserted")
        time.sleep(PAUSE)

    _print_summary("BACKFILL", len(stock_map), success, failed, total_rows)


def run_daily(engine: sa.Engine, stock_map: dict[str, int]) -> None:
    """
    Fetch the last 5 days for each stock and insert any missing rows.
    Designed to run every morning before market open.
    """
    today = date.today()
    log.info(f"MODE: daily — updating {len(stock_map)} stocks for {today}")
    log.info("─" * 55)

    total_rows = 0
    success    = 0
    skipped    = 0
    failed     = []

    for i, (symbol, stock_id) in enumerate(stock_map.items(), 1):
        last_date = get_last_date(engine, stock_id)

        # Skip if data is already current (last record is today or yesterday
        # and today is not a trading day)
        if last_date and last_date >= today - timedelta(days=3):
            log.info(f"[{i:>2}/{len(stock_map)}] {symbol:<20} already current ({last_date}) — skip")
            skipped += 1
            continue

        log.info(f"[{i:>2}/{len(stock_map)}] {symbol:<20} last={last_date} — fetching...")
        df = download_ohlcv(symbol, DAILY_PERIOD)

        if df is None:
            failed.append(symbol)
            continue

        inserted   = insert_ohlcv(engine, stock_id, df)
        total_rows += inserted
        success    += 1
        log.info(f"  inserted {inserted} new rows")
        time.sleep(PAUSE)

    log.info(f"\nDaily update complete — {success} updated, {skipped} skipped, {len(failed)} failed")
    if failed:
        log.warning(f"Failed: {failed}")


def run_single_symbol(engine: sa.Engine, stock_map: dict[str, int], symbol: str) -> None:
    """
    Download and insert data for one specific stock.
    Used for recovery or debugging a specific symbol.
    """
    symbol = symbol.upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"

    if symbol not in stock_map:
        log.error(f"{symbol} not found in stocks table. Run seed_stocks.py first.")
        return

    stock_id = stock_map[symbol]
    log.info(f"MODE: symbol — {symbol} (id={stock_id})")

    df = download_ohlcv(symbol, BACKFILL_PERIOD)
    if df is None:
        log.error(f"Download failed for {symbol}")
        return

    inserted = insert_ohlcv(engine, stock_id, df)
    log.info(f"Done — {len(df)} rows downloaded, {inserted} inserted into ohlcv_daily")


def run_migrate_csvs(engine: sa.Engine, stock_map: dict[str, int]) -> None:
    """
    Migrate existing Phase 1 CSVs from data/raw/ into the database.
    Finds every .csv in data/raw/, matches it to a stock symbol,
    and inserts the rows into ohlcv_daily.

    This preserves all the data you already downloaded in Phase 1.
    """
    raw_dir = CONFIG["paths"]["raw_data"]
    csvs    = list(raw_dir.glob("*.csv"))

    if not csvs:
        log.warning(f"No CSV files found in {raw_dir}")
        return

    log.info(f"MODE: migrate — found {len(csvs)} CSVs in {raw_dir}")
    log.info("─" * 55)

    total_rows = 0
    success    = 0
    skipped    = 0
    failed     = []

    for csv_path in sorted(csvs):
        # Map filename back to symbol: "RELIANCE.csv" → "RELIANCE.NS"
        symbol = csv_path.stem.upper() + ".NS"

        if symbol not in stock_map:
            log.warning(f"  {csv_path.name:<25} — no matching stock in DB, skipping")
            skipped += 1
            continue

        stock_id = stock_map[symbol]

        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            df = df.sort_index()
            df = df[["Open", "High", "Low", "Close", "Volume"]].round(2)
            df = df.dropna(subset=["Close"])
            df = df[df["Volume"] > 0]

            inserted   = insert_ohlcv(engine, stock_id, df)
            total_rows += inserted
            success    += 1
            log.info(f"  {csv_path.name:<25} {len(df)} rows  →  {inserted} inserted")

        except Exception as e:
            log.error(f"  {csv_path.name} — failed: {e}")
            failed.append(csv_path.name)

    _print_summary("MIGRATE", len(csvs), success, failed, total_rows)
    log.info("Your Phase 1 CSV data is now in PostgreSQL.")


# ─── Verification ─────────────────────────────────────────────────────────────

def verify_database(engine: sa.Engine) -> None:
    """Print a summary of what's currently in ohlcv_daily."""
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM ohlcv_daily")
        ).scalar()

        per_stock = conn.execute(text("""
            SELECT s.symbol, COUNT(*) as rows,
                   MIN(o.time)::date as earliest,
                   MAX(o.time)::date as latest
            FROM ohlcv_daily o
            JOIN stocks s ON s.id = o.stock_id
            GROUP BY s.symbol
            ORDER BY s.symbol
        """)).fetchall()

    print("\n" + "═" * 62)
    print(f"  ohlcv_daily — {total:,} total rows across {len(per_stock)} stocks")
    print("═" * 62)
    print(f"  {'Symbol':<20} {'Rows':>6}  {'From':<12}  {'To'}")
    print(f"  {'─'*20} {'─'*6}  {'─'*12}  {'─'*12}")
    for row in per_stock:
        print(f"  {row.symbol:<20} {row.rows:>6}  {str(row.earliest):<12}  {row.latest}")
    print("═" * 62 + "\n")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _print_summary(mode: str, total: int, success: int, failed: list, rows: int) -> None:
    print("\n" + "═" * 55)
    print(f"  {mode} SUMMARY")
    print("═" * 55)
    print(f"  Stocks attempted : {total}")
    print(f"  Succeeded        : {success}")
    print(f"  Failed           : {len(failed)}")
    print(f"  Total rows added : {rows:,}")
    if failed:
        print(f"  Failed symbols   : {failed}")
    print("═" * 55 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TradeQ price ingestor — writes OHLCV data to PostgreSQL"
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "daily", "symbol", "migrate", "verify"],
        default="daily",
        help=(
            "backfill = full history for all stocks | "
            "daily    = only missing recent days | "
            "symbol   = one specific stock | "
            "migrate  = import Phase 1 CSVs | "
            "verify   = show DB summary only"
        )
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Stock symbol for --mode symbol. e.g. RELIANCE or RELIANCE.NS"
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    engine = get_engine()

    log.info("TradeQ — Price Ingestor")
    log.info(f"Mode     : {args.mode}")
    log.info(f"Database : {os.getenv('DATABASE_URL')}")
    log.info("─" * 55)

    if args.mode == "verify":
        verify_database(engine)
        return

    stock_map = get_stock_map(engine)
    log.info(f"Active stocks in DB: {len(stock_map)}")

    start = time.time()

    if args.mode == "backfill":
        run_backfill(engine, stock_map)

    elif args.mode == "daily":
        run_daily(engine, stock_map)

    elif args.mode == "symbol":
        if not args.symbol:
            log.error("--symbol is required for mode=symbol")
            sys.exit(1)
        run_single_symbol(engine, stock_map, args.symbol)

    elif args.mode == "migrate":
        run_migrate_csvs(engine, stock_map)

    # Always show DB state after any write operation
    verify_database(engine)

    elapsed = round(time.time() - start, 1)
    log.info(f"Done in {elapsed}s")


if __name__ == "__main__":
    main()