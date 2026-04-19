"""
TradeQ AI — Phase 1
File: phase1/download_stocks.py

What this script does:
  - Downloads historical OHLCV data for all Nifty 50 stocks from Yahoo Finance
  - Saves each stock as a clean CSV file inside data/raw/
  - Prints a summary so you know exactly what got downloaded and what failed
  - Handles errors gracefully — one bad stock never crashes the whole run

How to run:
  python3 phase1/download_stocks.py

"""

import os
import time
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, date

# ─── Logging setup ────────────────────────────────────────────────────────────
# This makes every print statement include a timestamp — good habit from day one
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────
# All Nifty 50 stocks. Yahoo Finance needs ".NS" suffix for NSE-listed stocks.
# Change DOWNLOAD_PERIOD to "2y", "5y", or "max" when you want more history.

NIFTY_50_SYMBOLS = [
    "RELIANCE.NS",   "TCS.NS",        "HDFCBANK.NS",   "BHARTIARTL.NS",
    "ICICIBANK.NS",  "INFOSYS.NS",    "SBIN.NS",        "HINDUNILVR.NS",
    "ITC.NS",        "LT.NS",         "KOTAKBANK.NS",   "AXISBANK.NS",
    "BAJFINANCE.NS", "MARUTI.NS",     "HCLTECH.NS",     "ASIANPAINT.NS",
    "ADANIENT.NS",   "ADANIPORTS.NS", "ULTRACEMCO.NS",  "TITAN.NS",
    "WIPRO.NS",      "NTPC.NS",       "POWERGRID.NS",   "COALINDIA.NS",
    "SUNPHARMA.NS",  "BAJAJFINSV.NS", "ONGC.NS",        "M&M.NS",
    "JSWSTEEL.NS",   "TATAMOTORS.NS", "TATASTEEL.NS",   "HINDALCO.NS",
    "TECHM.NS",      "INDUSINDBK.NS", "CIPLA.NS",       "GRASIM.NS",
    "BRITANNIA.NS",  "DRREDDY.NS",    "DIVISLAB.NS",    "BPCL.NS",
    "TATACONSUM.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS",  "EICHERMOT.NS",
    "NESTLEIND.NS",  "BAJAJ-AUTO.NS", "SBILIFE.NS",     "HDFCLIFE.NS",
    "SHRIRAMFIN.NS", "TRENT.NS",
]

DOWNLOAD_PERIOD = "1y"       # How much history to download ("1y", "2y", "5y", "max")
OUTPUT_DIR      = "data/raw" # Where CSV files are saved
PAUSE_SECONDS   = 0.5        # Small pause between downloads — avoids rate limiting


# ─── Helper functions ─────────────────────────────────────────────────────────

def ensure_output_dir(path: str) -> None:
    """Create the output directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)
    log.info(f"Output directory ready: {path}/")


def download_single_stock(symbol: str, period: str) -> pd.DataFrame | None:
    """
    Download OHLCV data for one stock symbol from Yahoo Finance.

    Args:
        symbol: NSE ticker with .NS suffix e.g. "RELIANCE.NS"
        period: How much history e.g. "1y", "2y", "5y"

    Returns:
        A cleaned DataFrame if successful, None if the download failed.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=True)

        # history() returns empty DataFrame if symbol is wrong or delisted
        if df.empty:
            log.warning(f"  {symbol:<20} — no data returned (symbol may be wrong)")
            return None

        # Clean up: drop columns we don't need yet, reset index so Date is a column
        df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        df.index = df.index.tz_localize(None)  # Remove timezone info for clean CSVs
        df.index.name = "Date"
        df = df.round(2)                        # Round prices to 2 decimal places

        return df

    except Exception as e:
        log.error(f"  {symbol:<20} — download failed: {e}")
        return None


def save_stock_csv(df: pd.DataFrame, symbol: str, output_dir: str) -> str:
    """
    Save a stock's DataFrame as a CSV file.

    Args:
        df:         The DataFrame to save
        symbol:     Stock symbol (used to name the file)
        output_dir: Folder to save into

    Returns:
        The full path of the saved file.
    """
    # Strip ".NS" from filename: "RELIANCE.NS" → "RELIANCE.csv"
    clean_name = symbol.replace(".NS", "").replace(".", "_")
    filepath   = os.path.join(output_dir, f"{clean_name}.csv")
    df.to_csv(filepath)
    return filepath


def print_summary(results: list[dict]) -> None:
    """Print a clean download summary table at the end."""
    success = [r for r in results if r["status"] == "ok"]
    failed  = [r for r in results if r["status"] == "failed"]

    print("\n" + "═" * 60)
    print(f"  DOWNLOAD SUMMARY — {date.today()}")
    print("═" * 60)
    print(f"  Total stocks attempted : {len(results)}")
    print(f"  Successfully downloaded: {len(success)}")
    print(f"  Failed / no data       : {len(failed)}")
    print()

    if success:
        print("  Downloaded stocks:")
        for r in success:
            print(f"    {r['symbol']:<22} {r['rows']:>5} rows   {r['from']} → {r['to']}")

    if failed:
        print()
        print("  Failed stocks (check symbols):")
        for r in failed:
            print(f"    {r['symbol']}")

    print("═" * 60)
    print(f"  CSV files saved to: {OUTPUT_DIR}/")
    print("═" * 60 + "\n")


# ─── Main function ────────────────────────────────────────────────────────────

def main():
    log.info("TradeQ — Phase 1 Stock Downloader")
    log.info(f"Downloading {len(NIFTY_50_SYMBOLS)} Nifty 50 stocks | Period: {DOWNLOAD_PERIOD}")
    log.info("─" * 50)

    ensure_output_dir(OUTPUT_DIR)

    results = []

    for i, symbol in enumerate(NIFTY_50_SYMBOLS, start=1):
        log.info(f"[{i:>2}/{len(NIFTY_50_SYMBOLS)}] Downloading {symbol} ...")

        df = download_single_stock(symbol, DOWNLOAD_PERIOD)

        if df is not None:
            filepath = save_stock_csv(df, symbol, OUTPUT_DIR)
            results.append({
                "symbol": symbol,
                "status": "ok",
                "rows":   len(df),
                "from":   str(df.index.min().date()),
                "to":     str(df.index.max().date()),
                "file":   filepath,
            })
            log.info(f"  {symbol:<20} — {len(df)} rows saved → {filepath}")
        else:
            results.append({
                "symbol": symbol,
                "status": "failed",
            })

        # Small pause to be polite to Yahoo Finance API
        time.sleep(PAUSE_SECONDS)

    print_summary(results)

    # ── Quick sanity check: load one file back and print first 5 rows ──────
    log.info("Sanity check — loading RELIANCE.csv back from disk:")
    sample_path = os.path.join(OUTPUT_DIR, "RELIANCE.csv")
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path, index_col="Date", parse_dates=True)
        print(sample.tail(5).to_string())
        print()
        log.info(f"Shape: {sample.shape[0]} rows × {sample.shape[1]} columns")
        log.info(f"Columns: {list(sample.columns)}")
    else:
        log.warning("RELIANCE.csv not found — check if download succeeded.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = round(time.time() - start, 1)
    log.info(f"Done in {elapsed}s")