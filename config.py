"""
TradeQ — Central Configuration
File: config.py

This is the single source of truth for every setting in TradeQ.
No script should ever hardcode a path, symbol list, color, or parameter.
Everything lives here. Everything imports from here.

How to use in any script:
    from config import CONFIG

    symbols  = CONFIG["symbols"]["nifty50"]
    data_dir = CONFIG["paths"]["raw_data"]

Author: TradeQ Project
"""

import os
from pathlib import Path

# ─── Root directory ───────────────────────────────────────────────────────────
# This resolves to wherever config.py lives — the project root.
# All paths below are relative to this, so the project works on any machine.
ROOT = Path(__file__).parent.resolve()


CONFIG = {

    # ─── Project identity ─────────────────────────────────────────────────────
    "project": {
        "name":        "TradeQ",
        "version":     "0.1.0",
        "phase":       1,
        "description": "Real-time AI stock prediction and analysis engine for Indian markets",
        "market":      "NSE/BSE",
        "timezone":    "Asia/Kolkata",
    },

    # ─── File system paths ────────────────────────────────────────────────────
    # All paths are absolute, derived from ROOT.
    # Never hardcode "data/raw" — always use CONFIG["paths"]["raw_data"].
    "paths": {
        "root":         ROOT,
        "raw_data":     ROOT / "data" / "raw",
        "charts":       ROOT / "data" / "charts",
        "models":       ROOT / "data" / "models",
        "logs":         ROOT / "data" / "logs",
        "notebooks":    ROOT / "notebooks",
        "phase1":       ROOT / "phase1",
        "phase2":       ROOT / "phase2",
        "phase3":       ROOT / "phase3",
        "phase4":       ROOT / "phase4",
    },

    # ─── Stock universe ───────────────────────────────────────────────────────
    # Yahoo Finance requires ".NS" suffix for NSE-listed stocks.
    # "nifty50"  → what we use in Phase 1 and 2
    # "watchlist" → your personal shortlist to monitor closely
    "symbols": {
        "nifty50": [
            "RELIANCE.NS",   "TCS.NS",        "HDFCBANK.NS",   "BHARTIARTL.NS",
            "ICICIBANK.NS",  "INFOSYS.NS",     "SBIN.NS",       "HINDUNILVR.NS",
            "ITC.NS",        "LT.NS",          "KOTAKBANK.NS",  "AXISBANK.NS",
            "BAJFINANCE.NS", "MARUTI.NS",      "HCLTECH.NS",    "ASIANPAINT.NS",
            "ADANIENT.NS",   "ADANIPORTS.NS",  "ULTRACEMCO.NS", "TITAN.NS",
            "WIPRO.NS",      "NTPC.NS",        "POWERGRID.NS",  "COALINDIA.NS",
            "SUNPHARMA.NS",  "BAJAJFINSV.NS",  "ONGC.NS",       "M&M.NS",
            "JSWSTEEL.NS",   "TATAMOTORS.NS",  "TATASTEEL.NS",  "HINDALCO.NS",
            "TECHM.NS",      "INDUSINDBK.NS",  "CIPLA.NS",      "GRASIM.NS",
            "BRITANNIA.NS",  "DRREDDY.NS",     "DIVISLAB.NS",   "BPCL.NS",
            "TATACONSUM.NS", "APOLLOHOSP.NS",  "HEROMOTOCO.NS", "EICHERMOT.NS",
            "NESTLEIND.NS",  "BAJAJ-AUTO.NS",  "SBILIFE.NS",    "HDFCLIFE.NS",
            "SHRIRAMFIN.NS", "TRENT.NS",
        ],

        # Sector groupings — used in Phase 2 for sector-level feature engineering
        "sectors": {
            "banking":     ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
                            "AXISBANK.NS", "INDUSINDBK.NS"],
            "it":          ["TCS.NS", "INFOSYS.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
            "energy":      ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "NTPC.NS", "POWERGRID.NS",
                            "COALINDIA.NS"],
            "auto":        ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
                            "HEROMOTOCO.NS", "EICHERMOT.NS"],
            "pharma":      ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS"],
            "fmcg":        ["HINDUNILVR.NS", "ITC.NS", "BRITANNIA.NS", "NESTLEIND.NS",
                            "TATACONSUM.NS"],
            "metals":      ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS"],
            "finance":     ["BAJFINANCE.NS", "BAJAJFINSV.NS", "SBILIFE.NS", "HDFCLIFE.NS",
                            "SHRIRAMFIN.NS"],
            "infra":       ["LT.NS", "ADANIPORTS.NS", "ADANIENT.NS", "ULTRACEMCO.NS",
                            "GRASIM.NS"],
            "consumer":    ["ASIANPAINT.NS", "TITAN.NS", "TRENT.NS", "APOLLOHOSP.NS"],
        },

        # Your personal watchlist — edit freely
        "watchlist": [
            "RELIANCE.NS",
            "HDFCBANK.NS",
            "TCS.NS",
            "TATAMOTORS.NS",
            "SBIN.NS",
        ],

        # Index symbols used for macro feature engineering in Phase 2
        "indices": {
            "nifty50":     "^NSEI",
            "banknifty":   "^NSEBANK",
            "india_vix":   "^INDIAVIX",
            "sensex":      "^BSESN",
            "sp500":       "^GSPC",       # US market signal
            "nasdaq":      "^IXIC",       # US tech signal
            "crude_oil":   "CL=F",        # Crude oil futures
            "usd_inr":     "USDINR=X",    # Rupee/Dollar rate
            "gold":        "GC=F",        # Gold futures
        },
    },

    # ─── Data download settings ───────────────────────────────────────────────
    "download": {
        "default_period":   "1y",       # Used by download_stocks.py
        "training_period":  "5y",       # Used when building the ML training dataset
        "intraday_period":  "60d",      # Max yfinance allows for intraday data
        "intraday_interval":"5m",       # 5-minute candles for live Phase 4 signals
        "pause_seconds":    0.5,        # Pause between downloads — avoids rate limiting
        "max_retries":      3,          # Retry failed downloads this many times
        "timeout_seconds":  30,         # Give up on a single download after this
    },

    # ─── Technical indicator parameters ──────────────────────────────────────
    # These are the exact same values used in explore_stock.py and will be
    # used by the Phase 2 feature engine. Change here → changes everywhere.
    "indicators": {
        "ema_short":        20,         # Short EMA period (days)
        "ema_long":         50,         # Long EMA period (days)
        "ema_signal":       200,        # Long-term trend EMA
        "rsi_period":       14,         # RSI lookback period
        "atr_period":       14,         # Average True Range period
        "bb_period":        20,         # Bollinger Band period
        "bb_std":           2,          # Bollinger Band standard deviations
        "macd_fast":        12,         # MACD fast EMA
        "macd_slow":        26,         # MACD slow EMA
        "macd_signal":      9,          # MACD signal line
        "volume_ma":        20,         # Volume moving average period
        "rolling_52w":      252,        # Trading days in a year (52-week window)
        "volatility_window":20,         # Rolling volatility window
        "momentum_window":  10,         # Short-term momentum lookback
    },

    # ─── Chart settings ───────────────────────────────────────────────────────
    "charts": {
        "figsize_single":   (16, 9),    # Single stock chart
        "figsize_analysis": (18, 12),   # Analysis chart (explore_stock.py)
        "figsize_compare":  (20, 14),   # Comparison chart
        "dpi":              150,
        "background":       "#FAFAF8",
        "grid_color":       "#E8E6DF",
        "text_primary":     "#2C2C2A",
        "text_muted":       "#888780",
        # Candle colors
        "candle_up":        "#1D9E75",
        "candle_down":      "#D85A30",
        # Indicator colors
        "ema_short":        "#378ADD",
        "ema_long":         "#EF9F27",
        "volume_up":        "#9FE1CB",
        "volume_down":      "#F5C4B3",
        # 52-week markers
        "high_52w":         "#1D9E75",
        "low_52w":          "#D85A30",
        # Multi-stock palette (up to 15 stocks)
        "palette": [
            "#378ADD", "#1D9E75", "#D85A30", "#EF9F27", "#7F77DD",
            "#D4537E", "#639922", "#E24B4A", "#0F6E56", "#854F0B",
            "#185FA5", "#993C1D", "#3C3489", "#5F5E5A", "#A32D2D",
        ],
    },

    # ─── Market session schedule (IST) ───────────────────────────────────────
    # Used by Phase 4 scheduler to know when to run each service.
    "market": {
        "open_time":            "09:15",
        "close_time":           "15:30",
        "pre_open_start":       "09:00",
        "morning_scan_target":  "09:10",  # TradeQ must have predictions ready by this time
        "trading_days":         ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "exchange":             "NSE",
        "currency":             "INR",
        "currency_symbol":      "₹",
    },

    # ─── Logging settings ─────────────────────────────────────────────────────
    "logging": {
        "level":    "INFO",             # DEBUG / INFO / WARNING / ERROR
        "format":   "%(asctime)s  %(levelname)s  %(message)s",
        "datefmt":  "%H:%M:%S",
        "to_file":  False,              # Set True in Phase 2 to log to data/logs/
    },

    # ─── Phase 3 ML settings (defined now, used later) ───────────────────────
    # Keeping these here from day one means Phase 3 just reads config —
    # no hunting through code to find what parameters were used.
    "ml": {
        "target":               "beat_nifty_next_day",  # What the model predicts
        "target_threshold":     0.01,   # Stock must outperform Nifty by > 1% to be label=1
        "test_size_days":       63,     # Hold out last 63 trading days (~3 months) for testing
        "validation_splits":    5,      # Walk-forward cross-validation folds
        "min_training_rows":    500,    # Don't train if fewer than this many rows available
        "feature_count":        50,     # Total features going into the model
        "random_seed":          42,     # For reproducibility
        "model_version_prefix": "tradeq_v",
        # XGBoost hyperparameters (starting point — will be tuned)
        "xgb_params": {
            "n_estimators":     500,
            "max_depth":        6,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder":False,
            "eval_metric":      "logloss",
            "random_state":     42,
        },
    },
}


# ─── Auto-create all directories on import ────────────────────────────────────
# The moment any script imports config, all required folders exist.
# No script should ever call os.makedirs() for standard directories.
def _ensure_dirs() -> None:
    dirs_to_create = [
        CONFIG["paths"]["raw_data"],
        CONFIG["paths"]["charts"],
        CONFIG["paths"]["models"],
        CONFIG["paths"]["logs"],
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

_ensure_dirs()


# ─── Convenience helpers ──────────────────────────────────────────────────────

def get_symbols(group: str = "nifty50") -> list[str]:
    """
    Return the symbol list for a given group.

    Args:
        group: "nifty50", "watchlist", or any sector name
               e.g. "banking", "it", "pharma"

    Returns:
        List of ticker symbols with .NS suffix.

    Example:
        from config import get_symbols
        symbols = get_symbols("banking")
        # ["HDFCBANK.NS", "ICICIBANK.NS", ...]
    """
    if group in CONFIG["symbols"]:
        result = CONFIG["symbols"][group]
        # Handle both flat lists and dicts (sectors is a dict of lists)
        if isinstance(result, dict):
            raise ValueError(
                f"'{group}' is a dict of sectors. "
                f"Use get_symbols('banking') for a specific sector."
            )
        return result

    # Check sector names
    if group in CONFIG["symbols"]["sectors"]:
        return CONFIG["symbols"]["sectors"][group]

    valid = list(CONFIG["symbols"].keys()) + list(CONFIG["symbols"]["sectors"].keys())
    raise ValueError(f"Unknown symbol group '{group}'. Valid options: {valid}")


def get_raw_path(symbol: str) -> Path:
    """
    Return the full path to a stock's CSV file.

    Args:
        symbol: With or without .NS suffix. e.g. "RELIANCE" or "RELIANCE.NS"

    Returns:
        Path object pointing to data/raw/RELIANCE.csv

    Example:
        from config import get_raw_path
        path = get_raw_path("RELIANCE")
        df   = pd.read_csv(path, index_col="Date", parse_dates=True)
    """
    clean = symbol.upper().replace(".NS", "").replace(".", "_")
    return CONFIG["paths"]["raw_data"] / f"{clean}.csv"


def get_chart_path(symbol: str, suffix: str = "chart") -> Path:
    """
    Return the full path for saving a chart PNG.

    Args:
        symbol: Stock name e.g. "RELIANCE"
        suffix: "chart", "analysis", or "comparison"

    Returns:
        Path object e.g. data/charts/RELIANCE_chart.png

    Example:
        from config import get_chart_path
        fig.savefig(get_chart_path("RELIANCE", "analysis"), dpi=150)
    """
    clean = symbol.upper().replace(".NS", "").replace(".", "_")
    return CONFIG["paths"]["charts"] / f"{clean}_{suffix}.png"


def is_market_open() -> bool:
    """
    Check if the NSE market is currently open based on system time.
    Uses Asia/Kolkata timezone. Used by Phase 4 scheduler.

    Returns:
        True if currently within market hours on a weekday.
    """
    from datetime import datetime
    import zoneinfo
    now = datetime.now(zoneinfo.ZoneInfo(CONFIG["project"]["timezone"]))
    if now.strftime("%A") not in CONFIG["market"]["trading_days"]:
        return False
    open_h,  open_m  = map(int, CONFIG["market"]["open_time"].split(":"))
    close_h, close_m = map(int, CONFIG["market"]["close_time"].split(":"))
    market_open  = now.replace(hour=open_h,  minute=open_m,  second=0, microsecond=0)
    market_close = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    return market_open <= now <= market_close


# ─── Quick self-test ──────────────────────────────────────────────────────────
# Run this file directly to verify everything is set up correctly:
#   python3 config.py

if __name__ == "__main__":
    import json

    print("\n" + "═" * 55)
    print("  TradeQ — Config Self-Test")
    print("═" * 55)
    print(f"  Project  : {CONFIG['project']['name']} v{CONFIG['project']['version']}")
    print(f"  Phase    : {CONFIG['project']['phase']}")
    print(f"  Root     : {CONFIG['paths']['root']}")
    print()
    print(f"  Paths created:")
    for name, path in CONFIG["paths"].items():
        if name == "root":
            continue
        exists = "exists" if Path(path).exists() else "MISSING"
        print(f"    {name:<12} {path}  [{exists}]")
    print()
    print(f"  Nifty 50 symbols loaded : {len(CONFIG['symbols']['nifty50'])}")
    print(f"  Sectors defined         : {len(CONFIG['symbols']['sectors'])}")
    print(f"  Watchlist               : {CONFIG['symbols']['watchlist']}")
    print()
    print(f"  Helpers test:")
    print(f"    get_symbols('banking') → {get_symbols('banking')[:3]} ...")
    print(f"    get_raw_path('RELIANCE') → {get_raw_path('RELIANCE')}")
    print(f"    get_chart_path('TCS', 'analysis') → {get_chart_path('TCS', 'analysis')}")
    print(f"    is_market_open() → {is_market_open()}")
    print()
    print(f"  All indicators: {list(CONFIG['indicators'].keys())}")
    print("═" * 55)
    print("  Config loaded successfully. Import with:")
    print("  from config import CONFIG, get_symbols, get_raw_path")
    print("═" * 55 + "\n")