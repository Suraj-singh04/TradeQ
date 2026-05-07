"""
TradeQ — Phase 2
File: phase2/services/feature_engine/engine.py

What this service does:
  - Reads OHLCV data from ohlcv_daily for every active stock
  - Computes all 20 technical indicator features using pandas-ta
  - Computes 10 price action features (returns, gaps, streaks etc.)
  - Computes 3 volume features (spike ratio, OBV slope etc.)
  - Fills macro context columns from index data (Nifty, VIX, USD/INR)
  - Writes completed feature rows into features_daily
  - Fills the ML label column (did stock beat Nifty next day by > 1%?)
  - Safe to re-run — uses INSERT ON CONFLICT DO UPDATE

Modes:
  full     — compute features for all stocks for all available history
  daily    — compute features only for yesterday (run every evening at 4 PM)
  symbol   — compute features for one specific stock
  labels   — backfill the label column for all historical rows
  verify   — print a summary of what is in features_daily

How to run:
  python3 phase2/services/feature_engine/engine.py --mode full
  python3 phase2/services/feature_engine/engine.py --mode daily
  python3 phase2/services/feature_engine/engine.py --mode symbol --symbol RELIANCE.NS
  python3 phase2/services/feature_engine/engine.py --mode labels
  python3 phase2/services/feature_engine/engine.py --mode verify

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
import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv
import os

load_dotenv(ROOT / ".env")

# ── Try importing pandas_ta — give clear error if missing ─────────────────────
try:
    import pandas_ta as ta
except ImportError:
    print("pandas-ta not installed. Run: pip install pandas-ta")
    sys.exit(1)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config shortcuts ─────────────────────────────────────────────────────────
IND      = CONFIG["indicators"]
ML_CFG   = CONFIG["ml"]
NIFTY_SYM= CONFIG["symbols"]["indices"]["nifty50"]   # "^NSEI"
VIX_SYM  = CONFIG["symbols"]["indices"]["india_vix"]  # "^INDIAVIX"
USDINR   = CONFIG["symbols"]["indices"]["usd_inr"]    # "USDINR=X"
CRUDE    = CONFIG["symbols"]["indices"]["crude_oil"]   # "CL=F"


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
    if not rows:
        raise RuntimeError("stocks table is empty. Run seed_stocks.py first.")
    return {row.symbol: row.id for row in rows}


def load_ohlcv(engine: sa.Engine, stock_id: int, min_rows: int = 250) -> pd.DataFrame | None:
    """
    Load full OHLCV history for one stock from the database.
    Returns None if fewer than min_rows rows exist (not enough for indicators).
    """
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT time::date AS date, open, high, low, close, volume
                FROM ohlcv_daily
                WHERE stock_id = :sid
                ORDER BY time ASC
            """),
            conn,
            params={"sid": stock_id},
            parse_dates=["date"],
            index_col="date",
        )

    # DB returns lowercase column names; rename to Title-case to match compute_features
    df.columns = [c.capitalize() for c in df.columns]
    # Coerce all OHLCV columns to numeric (DB may return object dtype)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) < min_rows:
        log.warning(f"  stock_id={stock_id} only has {len(df)} rows — need {min_rows} minimum")
        return None

    return df


def load_macro_data(engine: sa.Engine) -> pd.DataFrame:
    """
    Load macro index data — Nifty return, VIX, USD/INR, Crude.
    These are stored in ohlcv_daily under special stock_ids.
    Falls back to yfinance download if not in DB.
    """
    import yfinance as yf

    macro = pd.DataFrame()
    indices = {
        "nifty_ret":    NIFTY_SYM,
        "india_vix":    VIX_SYM,
        "usd_inr_chg":  USDINR,
        "crude_chg":    CRUDE,
    }

    for col, symbol in indices.items():
        try:
            df = yf.download(symbol, period="5y", auto_adjust=True, progress=False)
            if df.empty:
                continue
            s = df["Close"].squeeze()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            if col in ("nifty_ret", "usd_inr_chg", "crude_chg"):
                s = s.pct_change() * 100   # Convert to daily % return
            macro[col] = s
        except Exception as e:
            log.warning(f"  Could not load macro data for {symbol}: {e}")

    macro.index = pd.to_datetime(macro.index).normalize()
    return macro.round(4)


# ─── Feature computation ──────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical and price action features for one stock.

    Args:
        df:    OHLCV DataFrame indexed by date
        macro: Macro data DataFrame (Nifty return, VIX etc.) indexed by date

    Returns:
        DataFrame where each row = one trading day's features.
        All column names match features_daily table columns exactly.
    """
    f = pd.DataFrame(index=df.index)

    # ── Technical indicators via pandas-ta ────────────────────────────────────
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # RSI
    rsi = ta.rsi(close, length=IND["rsi_period"])
    f["rsi_14"] = rsi

    # MACD
    macd_df = ta.macd(close,
                      fast=IND["macd_fast"],
                      slow=IND["macd_slow"],
                      signal=IND["macd_signal"])
    if macd_df is not None and not macd_df.empty:
        cols = macd_df.columns.tolist()
        f["macd"]        = macd_df.iloc[:, 0]   # MACD line
        f["macd_signal"] = macd_df.iloc[:, 2]   # Signal line
        f["macd_hist"]   = macd_df.iloc[:, 1]   # Histogram

    # Bollinger Bands
    bb = ta.bbands(close, length=IND["bb_period"], std=IND["bb_std"])
    if bb is not None and not bb.empty:
        f["bb_upper"] = bb.iloc[:, 2]   # Upper band
        f["bb_lower"] = bb.iloc[:, 0]   # Lower band
        # Compute true %B manually: (close - lower) / (upper - lower)
        bb_range = bb.iloc[:, 2] - bb.iloc[:, 0]
        f["bb_pct_b"] = ((close - bb.iloc[:, 0]) / bb_range.replace(0, float("nan"))).round(4)
        f["bb_width"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / close
    
    # EMAs
    f["ema_20"]  = ta.ema(close, length=IND["ema_short"])
    f["ema_50"]  = ta.ema(close, length=IND["ema_long"])
    f["ema_200"] = ta.ema(close, length=IND["ema_signal"])
    f["ema_cross_20_50"] = (f["ema_20"] / f["ema_50"]).round(4)

    # ATR
    atr = ta.atr(high, low, close, length=IND["atr_period"])
    f["atr_14"] = atr

    # ADX
    adx_df = ta.adx(high, low, close, length=IND["atr_period"])
    if adx_df is not None and not adx_df.empty:
        f["adx_14"] = adx_df.iloc[:, 0]

    # OBV
    obv = ta.obv(close, volume)
    f["obv"]       = obv
    f["obv_slope"] = obv.diff(5)    # 5-day slope of OBV

    # Stochastic
    stoch = ta.stoch(high, low, close)
    if stoch is not None and not stoch.empty:
        f["stoch_k"] = stoch.iloc[:, 0]
        f["stoch_d"] = stoch.iloc[:, 1]

    # CCI
    cci = ta.cci(high, low, close, length=20)
    f["cci_20"] = cci

    # Williams %R
    willr = ta.willr(high, low, close, length=IND["rsi_period"])
    f["williams_r"] = willr

    # VWAP deviation (close vs VWAP as %)
    vwap = ta.vwap(high, low, close, volume)
    if vwap is not None:
        f["vwap_dev"] = ((close - vwap) / vwap * 100).round(4)

    # ── Price action features ─────────────────────────────────────────────────
    prev_close = close.shift(1)

    f["daily_return"] = (close.pct_change() * 100).round(4)
    f["log_return"]   = (np.log(close / prev_close) * 100).round(4)
    f["body_pct"]     = ((close - df["Open"]).abs() / df["Open"] * 100).round(4)
    f["gap_pct"]      = ((df["Open"] - prev_close) / prev_close * 100).round(4)

    # Where does close sit within today's high-low range? (0 = at low, 1 = at high)
    day_range = (high - low).replace(0, np.nan)
    f["close_position"] = ((close - low) / day_range).round(4)

    # 52-week high/low distances
    rolling_52 = IND["rolling_52w"]
    f["high_52w_dist"] = ((close / high.rolling(rolling_52).max()) - 1).round(4)
    f["low_52w_dist"]  = ((close / low.rolling(rolling_52).min()) - 1).round(4)

    # Rolling 20-day annualised volatility
    f["volatility_20d"] = (
        f["daily_return"].rolling(IND["volatility_window"]).std() * np.sqrt(252)
    ).round(4)

    # Consecutive green / red candle streaks
    is_green = (close >= df["Open"]).astype(int)
    f["consec_green"] = _rolling_streak(is_green, 1)
    f["consec_red"]   = _rolling_streak(is_green, 0)

    # ── Volume features ───────────────────────────────────────────────────────
    vol_ma20          = volume.rolling(IND["volume_ma"]).mean()
    f["volume_ma_20"] = pd.to_numeric(vol_ma20, errors="coerce").round(0).astype("Int64")
    f["volume_spike"] = (volume / vol_ma20).round(3)

    # ── Merge macro features ──────────────────────────────────────────────────
    if not macro.empty:
        f = f.join(macro, how="left")

    # Relative strength = stock daily return - Nifty daily return
    if "nifty_ret" in f.columns:
        f["nifty_daily_ret"] = f["nifty_ret"].round(4)
        f["relative_strength"] = (f["daily_return"] - f["nifty_ret"]).round(4)
        f = f.drop(columns=["nifty_ret"], errors="ignore")

    if "usd_inr_chg" in f.columns:
        f["usd_inr_change"] = f["usd_inr_chg"].round(4)
        f = f.drop(columns=["usd_inr_chg"], errors="ignore")

    if "crude_chg" in f.columns:
        f["crude_change"] = f["crude_chg"].round(4)
        f = f.drop(columns=["crude_chg"], errors="ignore")

    # Drop rows where critical indicators are all NaN
    # (first ~200 rows before indicators have enough data)
    f = f.dropna(subset=["rsi_14", "ema_50"], how="any")

    return f


def _rolling_streak(series: pd.Series, value: int, max_streak: int = 20) -> pd.Series:
    """
    Count consecutive occurrences of `value` ending at each row.
    Used for consec_green and consec_red features.
    """
    streaks = []
    current = 0
    for v in series:
        if v == value:
            current = min(current + 1, max_streak)
        else:
            current = 0
        streaks.append(current)
    return pd.Series(streaks, index=series.index, dtype="Int16")


# ─── Label computation ────────────────────────────────────────────────────────

def compute_labels(df_features: pd.DataFrame, nifty_returns: pd.Series) -> pd.DataFrame:
    """
    Compute the ML target label for each row.

    Label = 1 if the stock's next-day return beats Nifty by > threshold%
    Label = 0 otherwise
    Label = NULL for the most recent row (future is unknown)

    Args:
        df_features:   Features DataFrame with daily_return column
        nifty_returns: Nifty 50 daily returns Series indexed by date

    Returns:
        df_features with next_day_return and label columns added.
    """
    threshold = ML_CFG["target_threshold"] * 100   # Convert 0.01 → 1.0 (%)

    # Shift returns back by 1 — "what happened the next day?"
    next_ret   = df_features["daily_return"].shift(-1)
    nifty_next = nifty_returns.reindex(df_features.index).shift(-1)

    df_features["next_day_return"] = next_ret.round(4)

    # Label: did stock beat Nifty by > threshold next day?
    outperformance = next_ret - nifty_next
    df_features["label"] = (outperformance > threshold).astype("Int8")

    # Last row: unknown (market hasn't happened yet) → NULL
    df_features.iloc[-1, df_features.columns.get_loc("label")] = pd.NA
    df_features.iloc[-1, df_features.columns.get_loc("next_day_return")] = pd.NA

    return df_features


# ─── Database write ───────────────────────────────────────────────────────────

UPSERT_SQL = text("""
    INSERT INTO features_daily (
        time, stock_id,
        rsi_14, macd, macd_signal, macd_hist,
        bb_upper, bb_lower, bb_pct_b, bb_width,
        ema_20, ema_50, ema_200, ema_cross_20_50,
        atr_14, adx_14, obv, obv_slope, vwap_dev,
        stoch_k, stoch_d, cci_20, williams_r,
        daily_return, log_return, body_pct, gap_pct,
        high_52w_dist, low_52w_dist, close_position,
        consec_green, consec_red, volatility_20d,
        volume_spike, volume_ma_20,
        nifty_daily_ret, india_vix, relative_strength,
        usd_inr_change, crude_change,
        next_day_return, label
    ) VALUES (
        :time, :stock_id,
        :rsi_14, :macd, :macd_signal, :macd_hist,
        :bb_upper, :bb_lower, :bb_pct_b, :bb_width,
        :ema_20, :ema_50, :ema_200, :ema_cross_20_50,
        :atr_14, :adx_14, :obv, :obv_slope, :vwap_dev,
        :stoch_k, :stoch_d, :cci_20, :williams_r,
        :daily_return, :log_return, :body_pct, :gap_pct,
        :high_52w_dist, :low_52w_dist, :close_position,
        :consec_green, :consec_red, :volatility_20d,
        :volume_spike, :volume_ma_20,
        :nifty_daily_ret, :india_vix, :relative_strength,
        :usd_inr_change, :crude_change,
        :next_day_return, :label
    )
    ON CONFLICT (time, stock_id) DO UPDATE SET
        rsi_14          = EXCLUDED.rsi_14,
        macd            = EXCLUDED.macd,
        macd_signal     = EXCLUDED.macd_signal,
        macd_hist       = EXCLUDED.macd_hist,
        ema_20          = EXCLUDED.ema_20,
        ema_50          = EXCLUDED.ema_50,
        ema_200         = EXCLUDED.ema_200,
        bb_pct_b        = EXCLUDED.bb_pct_b,
        volume_spike    = EXCLUDED.volume_spike,
        relative_strength = EXCLUDED.relative_strength,
        label           = EXCLUDED.label,
        next_day_return = EXCLUDED.next_day_return,
        updated_at      = NOW()
""")


def _safe_val(val):
    """Convert numpy/pandas NA types to Python None for SQLAlchemy."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        # Guard against values that would overflow any NUMERIC column
        if abs(v) > 999999:
            return None
        return v
    return val


def write_features(engine: sa.Engine, stock_id: int, features: pd.DataFrame) -> int:
    """
    Write feature rows to features_daily.
    Returns number of rows upserted.
    """
    rows = []
    for dt, row in features.iterrows():
        rows.append({
            "time":             datetime.combine(dt, datetime.min.time()),
            "stock_id":         stock_id,
            "rsi_14":           _safe_val(row.get("rsi_14")),
            "macd":             _safe_val(row.get("macd")),
            "macd_signal":      _safe_val(row.get("macd_signal")),
            "macd_hist":        _safe_val(row.get("macd_hist")),
            "bb_upper":         _safe_val(row.get("bb_upper")),
            "bb_lower":         _safe_val(row.get("bb_lower")),
            "bb_pct_b":         _safe_val(row.get("bb_pct_b")),
            "bb_width":         _safe_val(row.get("bb_width")),
            "ema_20":           _safe_val(row.get("ema_20")),
            "ema_50":           _safe_val(row.get("ema_50")),
            "ema_200":          _safe_val(row.get("ema_200")),
            "ema_cross_20_50":  _safe_val(row.get("ema_cross_20_50")),
            "atr_14":           _safe_val(row.get("atr_14")),
            "adx_14":           _safe_val(row.get("adx_14")),
            "obv":              _safe_val(row.get("obv")),
            "obv_slope":        _safe_val(row.get("obv_slope")),
            "vwap_dev":         _safe_val(row.get("vwap_dev")),
            "stoch_k":          _safe_val(row.get("stoch_k")),
            "stoch_d":          _safe_val(row.get("stoch_d")),
            "cci_20":           _safe_val(row.get("cci_20")),
            "williams_r":       _safe_val(row.get("williams_r")),
            "daily_return":     _safe_val(row.get("daily_return")),
            "log_return":       _safe_val(row.get("log_return")),
            "body_pct":         _safe_val(row.get("body_pct")),
            "gap_pct":          _safe_val(row.get("gap_pct")),
            "high_52w_dist":    _safe_val(row.get("high_52w_dist")),
            "low_52w_dist":     _safe_val(row.get("low_52w_dist")),
            "close_position":   _safe_val(row.get("close_position")),
            "consec_green":     _safe_val(row.get("consec_green")),
            "consec_red":       _safe_val(row.get("consec_red")),
            "volatility_20d":   _safe_val(row.get("volatility_20d")),
            "volume_spike":     _safe_val(row.get("volume_spike")),
            "volume_ma_20":     _safe_val(row.get("volume_ma_20")),
            "nifty_daily_ret":  _safe_val(row.get("nifty_daily_ret")),
            "india_vix":        _safe_val(row.get("india_vix")),
            "relative_strength":_safe_val(row.get("relative_strength")),
            "usd_inr_change":   _safe_val(row.get("usd_inr_change")),
            "crude_change":     _safe_val(row.get("crude_change")),
            "next_day_return":  _safe_val(row.get("next_day_return")),
            "label":            _safe_val(row.get("label")),
        })

    if not rows:
        return 0

    with engine.begin() as conn:
        result = conn.execute(UPSERT_SQL, rows)

    return result.rowcount


# ─── Modes ────────────────────────────────────────────────────────────────────

def run_full(engine: sa.Engine, stock_map: dict[str, int], macro: pd.DataFrame) -> None:
    """Compute features for all stocks across full history."""
    log.info(f"MODE: full — processing {len(stock_map)} stocks")
    log.info("Downloading macro data first...")

    nifty_ret = macro["nifty_ret"] if "nifty_ret" in macro.columns else pd.Series(dtype=float)

    total_rows = 0
    success    = 0
    failed     = []

    for i, (symbol, stock_id) in enumerate(stock_map.items(), 1):
        log.info(f"[{i:>2}/{len(stock_map)}] {symbol}")

        df = load_ohlcv(engine, stock_id)
        if df is None:
            failed.append(symbol)
            continue

        try:
            features = compute_features(df, macro)
            features = compute_labels(features, nifty_ret)
            inserted = write_features(engine, stock_id, features)
            total_rows += inserted
            success    += 1
            log.info(f"  {symbol:<20} {len(features)} feature rows  →  {inserted} written")
        except Exception as e:
            log.error(f"  {symbol} — feature computation failed: {e}")
            failed.append(symbol)

    print("\n" + "═" * 55)
    print(f"  FULL RUN SUMMARY")
    print("═" * 55)
    print(f"  Stocks processed : {success}/{len(stock_map)}")
    print(f"  Total rows       : {total_rows:,}")
    print(f"  Failed           : {failed or 'none'}")
    print("═" * 55 + "\n")


def run_daily(engine: sa.Engine, stock_map: dict[str, int], macro: pd.DataFrame) -> None:
    """Compute features for yesterday only. Run every evening after market close."""
    yesterday = date.today() - timedelta(days=1)
    log.info(f"MODE: daily — computing features for {yesterday}")

    nifty_ret = macro["nifty_ret"] if "nifty_ret" in macro.columns else pd.Series(dtype=float)
    success = 0

    for symbol, stock_id in stock_map.items():
        df = load_ohlcv(engine, stock_id)
        if df is None:
            continue
        try:
            features = compute_features(df, macro)
            features = compute_labels(features, nifty_ret)
            # Only write yesterday's row
            if yesterday in features.index:
                day_features = features.loc[[yesterday]]
                inserted = write_features(engine, stock_id, day_features)
                if inserted:
                    success += 1
                    log.info(f"  {symbol:<20} features written for {yesterday}")
        except Exception as e:
            log.error(f"  {symbol} — {e}")

    log.info(f"Daily run complete — {success} stocks updated")


def run_single_symbol(
    engine: sa.Engine, stock_map: dict[str, int],
    macro: pd.DataFrame, symbol: str
) -> None:
    """Compute features for one stock."""
    symbol = symbol.upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"

    if symbol not in stock_map:
        log.error(f"{symbol} not in stocks table.")
        return

    stock_id  = stock_map[symbol]
    nifty_ret = macro["nifty_ret"] if "nifty_ret" in macro.columns else pd.Series(dtype=float)

    df = load_ohlcv(engine, stock_id)
    if df is None:
        return

    features = compute_features(df, macro)
    features = compute_labels(features, nifty_ret)
    inserted = write_features(engine, stock_id, features)
    log.info(f"{symbol} — {len(features)} rows computed, {inserted} written to features_daily")


def run_verify(engine: sa.Engine) -> None:
    """Print a summary of what is in features_daily."""
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM features_daily")
        ).scalar()

        label_dist = conn.execute(text("""
            SELECT label, COUNT(*) as cnt
            FROM features_daily
            WHERE label IS NOT NULL
            GROUP BY label ORDER BY label
        """)).fetchall()

        per_stock = conn.execute(text("""
            SELECT s.symbol,
                   COUNT(*) as rows,
                   MIN(f.time)::date as earliest,
                   MAX(f.time)::date as latest,
                   ROUND(AVG(f.rsi_14)::numeric, 1) as avg_rsi,
                   COUNT(f.label) as labelled_rows
            FROM features_daily f
            JOIN stocks s ON s.id = f.stock_id
            GROUP BY s.symbol
            ORDER BY s.symbol
        """)).fetchall()

    print("\n" + "═" * 72)
    print(f"  features_daily — {total:,} total rows across {len(per_stock)} stocks")
    print("═" * 72)
    print(f"  {'Symbol':<20} {'Rows':>6}  {'From':<12}  {'To':<12}  {'AvgRSI':>7}  {'Labelled':>9}")
    print(f"  {'─'*20} {'─'*6}  {'─'*12}  {'─'*12}  {'─'*7}  {'─'*9}")
    for row in per_stock:
        print(
            f"  {row.symbol:<20} {row.rows:>6}  "
            f"{str(row.earliest):<12}  {str(row.latest):<12}  "
            f"{str(row.avg_rsi):>7}  {row.labelled_rows:>9}"
        )

    if label_dist:
        total_labelled = sum(r.cnt for r in label_dist)
        print(f"\n  Label distribution (total labelled rows: {total_labelled:,})")
        for row in label_dist:
            label_name = "Buy (beat Nifty)" if row.label == 1 else "Hold (did not beat)"
            pct = row.cnt / total_labelled * 100
            print(f"    Label {row.label} — {label_name:<22} {row.cnt:>7,} rows ({pct:.1f}%)")

    print("═" * 72 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TradeQ feature engine — computes ML features from OHLCV data"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "daily", "symbol", "labels", "verify"],
        default="daily",
    )
    parser.add_argument("--symbol", type=str, default=None)
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    engine = get_engine()

    log.info("TradeQ — Feature Engine")
    log.info(f"Mode: {args.mode}")
    log.info("─" * 55)

    if args.mode == "verify":
        run_verify(engine)
        return

    stock_map = get_stock_map(engine)
    log.info(f"Active stocks: {len(stock_map)}")

    log.info("Loading macro data (Nifty, VIX, USD/INR, Crude)...")
    macro = load_macro_data(engine)
    log.info(f"Macro data loaded: {len(macro)} rows, columns: {list(macro.columns)}")

    start = time.time()

    if args.mode == "full":
        run_full(engine, stock_map, macro)

    elif args.mode == "daily":
        run_daily(engine, stock_map, macro)

    elif args.mode == "symbol":
        if not args.symbol:
            log.error("--symbol required for mode=symbol")
            sys.exit(1)
        run_single_symbol(engine, stock_map, macro, args.symbol)

    run_verify(engine)
    log.info(f"Done in {round(time.time() - start, 1)}s")


if __name__ == "__main__":
    main()