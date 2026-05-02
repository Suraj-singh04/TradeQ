"""
TradeQ — Phase 2
File: phase2/database/seeds/seed_stocks.py

What this script does:
  - Populates the stocks table with all 50 Nifty companies
  - Fetches real company names and sector info from yfinance
  - Safe to re-run — uses INSERT ... ON CONFLICT DO UPDATE
    so existing rows are updated, not duplicated

How to run (once, before price_ingestor):
  python3 phase2/database/seeds/seed_stocks.py

Author: TradeQ Project
"""

import sys
import time
import logging
from pathlib import Path

# ── Make sure project root is on the path so config.py is importable ─────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import CONFIG, get_symbols
import yfinance as yf
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


# ─── Nifty 50 — fallback metadata ─────────────────────────────────────────────
# yfinance is used as primary source for names/sectors.
# This dict is the fallback if yfinance returns incomplete data.
# Sector names match CONFIG["symbols"]["sectors"] keys exactly.

STOCK_METADATA = {
    "RELIANCE.NS":   ("Reliance Industries Ltd",          "energy",    "Oil & Gas Refining"),
    "TCS.NS":        ("Tata Consultancy Services Ltd",     "it",        "IT Services"),
    "HDFCBANK.NS":   ("HDFC Bank Ltd",                    "banking",   "Private Sector Bank"),
    "BHARTIARTL.NS": ("Bharti Airtel Ltd",                "telecom",   "Telecom Services"),
    "ICICIBANK.NS":  ("ICICI Bank Ltd",                   "banking",   "Private Sector Bank"),
    "INFOSYS.NS":    ("Infosys Ltd",                      "it",        "IT Services"),
    "SBIN.NS":       ("State Bank of India",              "banking",   "Public Sector Bank"),
    "HINDUNILVR.NS": ("Hindustan Unilever Ltd",           "fmcg",      "FMCG"),
    "ITC.NS":        ("ITC Ltd",                          "fmcg",      "Diversified FMCG"),
    "LT.NS":         ("Larsen & Toubro Ltd",              "infra",     "Engineering & Construction"),
    "KOTAKBANK.NS":  ("Kotak Mahindra Bank Ltd",          "banking",   "Private Sector Bank"),
    "AXISBANK.NS":   ("Axis Bank Ltd",                    "banking",   "Private Sector Bank"),
    "BAJFINANCE.NS": ("Bajaj Finance Ltd",                "finance",   "NBFC"),
    "MARUTI.NS":     ("Maruti Suzuki India Ltd",          "auto",      "Passenger Vehicles"),
    "HCLTECH.NS":    ("HCL Technologies Ltd",             "it",        "IT Services"),
    "ASIANPAINT.NS": ("Asian Paints Ltd",                 "consumer",  "Paints"),
    "ADANIENT.NS":   ("Adani Enterprises Ltd",            "infra",     "Diversified"),
    "ADANIPORTS.NS": ("Adani Ports & SEZ Ltd",            "infra",     "Ports & Logistics"),
    "ULTRACEMCO.NS": ("UltraTech Cement Ltd",             "infra",     "Cement"),
    "TITAN.NS":      ("Titan Company Ltd",                "consumer",  "Jewellery & Watches"),
    "WIPRO.NS":      ("Wipro Ltd",                        "it",        "IT Services"),
    "NTPC.NS":       ("NTPC Ltd",                         "energy",    "Power Generation"),
    "POWERGRID.NS":  ("Power Grid Corp of India Ltd",     "energy",    "Power Transmission"),
    "COALINDIA.NS":  ("Coal India Ltd",                   "energy",    "Coal Mining"),
    "SUNPHARMA.NS":  ("Sun Pharmaceutical Industries Ltd","pharma",    "Pharmaceuticals"),
    "BAJAJFINSV.NS": ("Bajaj Finserv Ltd",                "finance",   "Financial Services"),
    "ONGC.NS":       ("Oil & Natural Gas Corp Ltd",       "energy",    "Oil & Gas Exploration"),
    "M&M.NS":        ("Mahindra & Mahindra Ltd",          "auto",      "SUVs & Tractors"),
    "JSWSTEEL.NS":   ("JSW Steel Ltd",                    "metals",    "Steel"),
    "TATAMOTORS.NS": ("Tata Motors Ltd",                  "auto",      "Commercial & Passenger Vehicles"),
    "TATASTEEL.NS":  ("Tata Steel Ltd",                   "metals",    "Steel"),
    "HINDALCO.NS":   ("Hindalco Industries Ltd",          "metals",    "Aluminium & Copper"),
    "TECHM.NS":      ("Tech Mahindra Ltd",                "it",        "IT Services"),
    "INDUSINDBK.NS": ("IndusInd Bank Ltd",                "banking",   "Private Sector Bank"),
    "CIPLA.NS":      ("Cipla Ltd",                        "pharma",    "Pharmaceuticals"),
    "GRASIM.NS":     ("Grasim Industries Ltd",            "infra",     "Cement & VSF"),
    "BRITANNIA.NS":  ("Britannia Industries Ltd",         "fmcg",      "Bakery & Dairy"),
    "DRREDDY.NS":    ("Dr Reddys Laboratories Ltd",       "pharma",    "Pharmaceuticals"),
    "DIVISLAB.NS":   ("Divis Laboratories Ltd",           "pharma",    "API Manufacturing"),
    "BPCL.NS":       ("Bharat Petroleum Corp Ltd",        "energy",    "Oil Refining & Marketing"),
    "TATACONSUM.NS": ("Tata Consumer Products Ltd",       "fmcg",      "Beverages & Foods"),
    "APOLLOHOSP.NS": ("Apollo Hospitals Enterprise Ltd",  "consumer",  "Healthcare"),
    "HEROMOTOCO.NS": ("Hero MotoCorp Ltd",                "auto",      "Two-Wheelers"),
    "EICHERMOT.NS":  ("Eicher Motors Ltd",                "auto",      "Motorcycles & Trucks"),
    "NESTLEIND.NS":  ("Nestle India Ltd",                 "fmcg",      "Food & Beverages"),
    "BAJAJ-AUTO.NS": ("Bajaj Auto Ltd",                   "auto",      "Two & Three Wheelers"),
    "SBILIFE.NS":    ("SBI Life Insurance Co Ltd",        "finance",   "Life Insurance"),
    "HDFCLIFE.NS":   ("HDFC Life Insurance Co Ltd",       "finance",   "Life Insurance"),
    "SHRIRAMFIN.NS": ("Shriram Finance Ltd",              "finance",   "NBFC"),
    "TRENT.NS":      ("Trent Ltd",                        "consumer",  "Retail"),
}


# ─── Database connection ──────────────────────────────────────────────────────

def get_engine() -> sa.Engine:
    """Create SQLAlchemy engine from DATABASE_URL in .env"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError(
            "DATABASE_URL not found in .env\n"
            "Expected: postgresql://tradeq:tradeq_secret@localhost:5432/tradeq_db"
        )
    return sa.create_engine(db_url, pool_pre_ping=True)


# ─── Fetch live data from yfinance ────────────────────────────────────────────

def fetch_yfinance_info(symbol: str) -> dict:
    """
    Fetch company name, sector and market cap from yfinance.
    Falls back to STOCK_METADATA if yfinance returns incomplete data.

    Args:
        symbol: NSE ticker e.g. "RELIANCE.NS"

    Returns:
        Dict with keys: name, sector, industry, market_cap_cr
    """
    fallback_name, fallback_sector, fallback_industry = STOCK_METADATA.get(
        symbol, (symbol.replace(".NS", ""), "unknown", "unknown")
    )

    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info

        name     = info.get("longName") or info.get("shortName") or fallback_name
        sector   = fallback_sector   # Always use our sector mapping — yfinance uses US sector names
        industry = info.get("industry") or fallback_industry

        # marketCap is in INR for NSE stocks — convert to crores (1 cr = 10M)
        market_cap    = info.get("marketCap") or 0
        market_cap_cr = int(market_cap / 10_000_000) if market_cap else None

        return {
            "name":          name.strip(),
            "sector":        sector,
            "industry":      industry,
            "market_cap_cr": market_cap_cr,
        }

    except Exception as e:
        log.warning(f"  yfinance failed for {symbol}: {e} — using fallback metadata")
        return {
            "name":          fallback_name,
            "sector":        fallback_sector,
            "industry":      fallback_industry,
            "market_cap_cr": None,
        }


# ─── Seed function ────────────────────────────────────────────────────────────

def seed_stocks(engine: sa.Engine) -> None:
    """
    Insert or update all Nifty 50 stocks in the stocks table.
    Uses ON CONFLICT DO UPDATE so re-running is always safe.
    """
    symbols = get_symbols("nifty50")
    log.info(f"Seeding {len(symbols)} stocks into the database...")
    log.info("─" * 50)

    upsert_sql = text("""
        INSERT INTO stocks (symbol, name, sector, industry, market_cap_cr, is_active, updated_at)
        VALUES (:symbol, :name, :sector, :industry, :market_cap_cr, TRUE, NOW())
        ON CONFLICT (symbol) DO UPDATE SET
            name          = EXCLUDED.name,
            sector        = EXCLUDED.sector,
            industry      = EXCLUDED.industry,
            market_cap_cr = EXCLUDED.market_cap_cr,
            is_active     = TRUE,
            updated_at    = NOW()
    """)

    success = 0
    failed  = []

    with engine.begin() as conn:
        for i, symbol in enumerate(symbols, 1):
            log.info(f"[{i:>2}/{len(symbols)}] {symbol} ...")
            info = fetch_yfinance_info(symbol)

            try:
                conn.execute(upsert_sql, {
                    "symbol":        symbol,
                    "name":          info["name"],
                    "sector":        info["sector"],
                    "industry":      info["industry"],
                    "market_cap_cr": info["market_cap_cr"],
                })
                log.info(
                    f"  OK  {info['name'][:45]:<45} "
                    f"sector={info['sector']:<10} "
                    f"mcap=₹{info['market_cap_cr']:,}cr"
                    if info["market_cap_cr"] else
                    f"  OK  {info['name'][:45]:<45} sector={info['sector']}"
                )
                success += 1

            except Exception as e:
                log.error(f"  DB insert failed for {symbol}: {e}")
                failed.append(symbol)

            time.sleep(0.3)  # be polite to yfinance

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print(f"  SEED SUMMARY")
    print("═" * 55)
    print(f"  Inserted / updated : {success}")
    print(f"  Failed             : {len(failed)}")
    if failed:
        print(f"  Failed symbols     : {failed}")
    print("═" * 55)


def verify_seed(engine: sa.Engine) -> None:
    """Print what's now in the stocks table so you can confirm it looks right."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT symbol, name, sector, market_cap_cr "
            "FROM stocks ORDER BY sector, symbol"
        )).fetchall()

    print(f"\n  {len(rows)} stocks in database:\n")
    current_sector = None
    for row in rows:
        if row.sector != current_sector:
            current_sector = row.sector
            print(f"\n  [{current_sector.upper()}]")
        mcap = f"₹{row.market_cap_cr:,}cr" if row.market_cap_cr else "mcap unknown"
        print(f"    {row.symbol:<20} {row.name[:40]:<40} {mcap}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("TradeQ — Phase 2 Stock Seed")
    log.info(f"Database: {os.getenv('DATABASE_URL', 'NOT SET — check .env')}")

    engine = get_engine()
    seed_stocks(engine)
    verify_seed(engine)

    log.info("Seed complete. stocks table is ready for price_ingestor.")


if __name__ == "__main__":
    main()