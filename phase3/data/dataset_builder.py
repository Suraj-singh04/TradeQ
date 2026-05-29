"""
TradeQ — Phase 3
File: phase3/data/dataset_builder.py

What this does:
  - Pulls features_daily from PostgreSQL into a clean DataFrame
  - Drops columns that are too sparse (> 30% nulls)
  - Fills remaining nulls with column median (per stock)
  - Removes data leakage columns (next_day_return, label metadata)
  - Produces walk-forward train/test splits
  - Saves the cleaned dataset to data/models/dataset.parquet

Walk-forward split logic:
  Train: everything before cutoff date
  Test:  everything from cutoff date onward
  Default cutoff: last 252 trading days (~1 year) reserved for testing

How to run:
  python3 phase3/data/dataset_builder.py
  python3 phase3/data/dataset_builder.py --test-days 126  # 6-month test set

Author: TradeQ Project
"""

import sys
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import CONFIG
import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv
import os

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

# Columns that must never be used as features
# next_day_return IS the label source — using it would be leakage
# time, stock_id, created_at, updated_at are metadata
EXCLUDE_COLS = {
    "time", "stock_id", "label", "next_day_return",
    "created_at", "updated_at", "symbol", "name", "sector",
}

# Drop any feature column where more than this % of values are NULL
NULL_THRESHOLD = 0.30

# Default test set size in trading days
DEFAULT_TEST_DAYS = 252   # ~1 year


# ─── Database ─────────────────────────────────────────────────────────────────

def get_engine() -> sa.Engine:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL not set in .env")
    return sa.create_engine(db_url, pool_pre_ping=True)


def load_raw_features(engine: sa.Engine) -> pd.DataFrame:
    """
    Load all labelled rows from features_daily joined with stock symbol.
    Only rows where label IS NOT NULL are included — unlabelled rows
    (today's data where next day hasn't happened) are excluded.
    """
    log.info("Loading features_daily from database...")

    query = text("""
        SELECT
            f.*,
            s.symbol,
            s.sector
        FROM features_daily f
        JOIN stocks s ON s.id = f.stock_id
        WHERE f.label IS NOT NULL
        ORDER BY f.time ASC, f.stock_id ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["time"])

    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    log.info(f"Date range: {df['time'].min().date()} → {df['time'].max().date()}")
    log.info(f"Stocks: {df['stock_id'].nunique()}")

    return df


# ─── Cleaning ─────────────────────────────────────────────────────────────────

def drop_leakage_and_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that must never be features."""
    cols_to_drop = [c for c in EXCLUDE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.info(f"Dropped {len(cols_to_drop)} metadata/leakage columns: {cols_to_drop}")
    return df


def drop_sparse_columns(df: pd.DataFrame, threshold: float = NULL_THRESHOLD) -> pd.DataFrame:
    """
    Drop feature columns where more than `threshold` fraction of values are NULL.
    These columns don't have enough data to be useful and would require
    excessive imputation that could distort the model.
    """
    null_rates = df.isnull().mean()
    sparse_cols = null_rates[null_rates > threshold].index.tolist()

    if sparse_cols:
        log.info(f"Dropping {len(sparse_cols)} sparse columns (>{threshold*100:.0f}% null):")
        for col in sparse_cols:
            log.info(f"  {col:<30} {null_rates[col]*100:.1f}% null")
        df = df.drop(columns=sparse_cols)
    else:
        log.info("No sparse columns found — all columns within null threshold")

    return df


def fill_nulls(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Fill remaining nulls with the column median.
    Using median (not mean) because financial features often have outliers
    that would skew the mean. Fill per-column globally (not per-stock)
    so the model sees consistent imputation at inference time.
    """
    null_counts = df[feature_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if len(cols_with_nulls) > 0:
        log.info(f"Filling nulls in {len(cols_with_nulls)} columns with median:")
        for col, count in cols_with_nulls.items():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.info(f"  {col:<30} {count:>5} nulls → filled with {median_val:.4f}")
    else:
        log.info("No remaining nulls after sparse column removal")

    return df


# ─── Feature selection ────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of columns to use as model features.
    Excludes metadata, label, and non-numeric columns.
    """
    # Keep only numeric columns that aren't excluded
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    return feature_cols


def analyze_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Print a summary of all features — useful for understanding what
    the model will train on before actually training.
    """
    stats = []
    for col in feature_cols:
        stats.append({
            "feature":   col,
            "null_pct":  round(df[col].isnull().mean() * 100, 1),
            "mean":      round(df[col].mean(), 4),
            "std":       round(df[col].std(), 4),
            "min":       round(df[col].min(), 4),
            "max":       round(df[col].max(), 4),
        })
    return pd.DataFrame(stats)


# ─── Walk-forward split ───────────────────────────────────────────────────────

def walk_forward_split(
    df: pd.DataFrame,
    test_days: int = DEFAULT_TEST_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets using walk-forward logic.

    CRITICAL: Never use random shuffle for financial time-series.
    The test set must always be strictly AFTER the training set.
    Otherwise the model learns from future data (leakage) and produces
    fake accuracy scores that collapse in live trading.

    Args:
        df:        Full cleaned DataFrame sorted by time ASC
        test_days: Number of most recent trading days to reserve for testing

    Returns:
        (train_df, test_df) — no overlap, test is always after train
    """
    # Get unique trading dates sorted chronologically
    dates = sorted(df["time"].dt.date.unique())
    total_days = len(dates)

    if test_days >= total_days:
        raise ValueError(
            f"test_days ({test_days}) >= total trading days ({total_days}). "
            f"Reduce test_days."
        )

    cutoff_date = dates[-test_days]
    log.info(f"Walk-forward split:")
    log.info(f"  Train: {dates[0]} → {dates[-test_days-1]}  ({total_days - test_days} days)")
    log.info(f"  Test : {cutoff_date} → {dates[-1]}  ({test_days} days)")

    train_df = df[df["time"].dt.date < cutoff_date].copy()
    test_df  = df[df["time"].dt.date >= cutoff_date].copy()

    log.info(f"  Train rows: {len(train_df):,}  ({train_df['stock_id'].nunique()} stocks)")
    log.info(f"  Test rows : {len(test_df):,}  ({test_df['stock_id'].nunique()} stocks)")

    return train_df, test_df


def make_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and label vector y from a DataFrame.

    Args:
        df:           DataFrame with feature columns and label column
        feature_cols: List of column names to use as features

    Returns:
        (X, y) where X is features DataFrame and y is label Series
    """
    X = df[feature_cols].copy()
    y = df["label"].astype(int).copy()
    return X, y


# ─── Save / load ──────────────────────────────────────────────────────────────

def save_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Path:
    """
    Save the cleaned dataset and metadata to data/models/.
    Parquet format is used — it preserves dtypes and is fast to load.
    """
    models_dir = CONFIG["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    # Full cleaned dataset
    dataset_path = models_dir / "dataset.parquet"
    df_save = df.copy()
    # Add label back for saving (was kept in df throughout)
    df_save.to_parquet(dataset_path, index=False)

    # Feature column list
    import json
    meta = {
        "feature_cols":   feature_cols,
        "n_features":     len(feature_cols),
        "n_rows":         len(df),
        "n_train":        len(train_df),
        "n_test":         len(test_df),
        "train_from":     str(train_df["time"].min().date()),
        "train_to":       str(train_df["time"].max().date()),
        "test_from":      str(test_df["time"].min().date()),
        "test_to":        str(test_df["time"].max().date()),
        "buy_rate":       round(df["label"].mean() * 100, 2),
        "created_at":     datetime.now().isoformat(),
    }
    meta_path = models_dir / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Dataset saved → {dataset_path}")
    log.info(f"Metadata saved → {meta_path}")

    return dataset_path


def load_dataset(models_dir: Path = None) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Load the saved dataset and metadata.
    Used by trainer.py and evaluator.py.

    Returns:
        (df, feature_cols, meta)
    """
    import json
    if models_dir is None:
        models_dir = CONFIG["paths"]["models"]

    dataset_path = models_dir / "dataset.parquet"
    meta_path    = models_dir / "dataset_meta.json"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run dataset_builder.py first."
        )

    df   = pd.read_parquet(dataset_path)
    with open(meta_path) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    log.info(f"Loaded dataset: {len(df):,} rows, {len(feature_cols)} features")

    return df, feature_cols, meta


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_dataset(test_days: int = DEFAULT_TEST_DAYS) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    """
    Full dataset build pipeline. Returns everything needed for training.

    Returns:
        (df, feature_cols, train_df, test_df)
    """
    engine = get_engine()

    # 1. Load raw data
    df = load_raw_features(engine)

    # 2. Drop leakage and metadata columns
    #    Keep time, label, stock_id in df for splitting
    df_clean = drop_leakage_and_metadata(df.copy())

    # Add label, time, stock_id back for splitting
    df_clean["label"]    = df["label"].values
    df_clean["time"]     = df["time"].values
    df_clean["stock_id"] = df["stock_id"].values

    # 3. Identify feature columns
    feature_cols = get_feature_columns(df_clean)
    log.info(f"Initial feature count: {len(feature_cols)}")

    # 4. Drop sparse columns
    df_features = df_clean[feature_cols + ["label", "time", "stock_id"]].copy()
    # Re-identify after potential drops
    df_features_clean = drop_sparse_columns(
        df_features.drop(columns=["label", "time", "stock_id"]),
        threshold=NULL_THRESHOLD,
    )
    # Restore label, time, stock_id
    df_features_clean["label"]    = df_features["label"].values
    df_features_clean["time"]     = df_features["time"].values
    df_features_clean["stock_id"] = df_features["stock_id"].values

    # 5. Update feature list after sparse removal
    feature_cols = get_feature_columns(df_features_clean)
    log.info(f"Features after sparse removal: {len(feature_cols)}")

    # 6. Fill remaining nulls
    df_features_clean = fill_nulls(df_features_clean, feature_cols)

    # 7. Final null check
    remaining_nulls = df_features_clean[feature_cols].isnull().sum().sum()
    if remaining_nulls > 0:
        log.warning(f"WARNING: {remaining_nulls} nulls remain after filling — investigate")
    else:
        log.info("Null check passed — zero nulls in feature columns")

    # 8. Walk-forward split
    train_df, test_df = walk_forward_split(df_features_clean, test_days)

    # 9. Print label distribution
    train_buy = train_df["label"].mean() * 100
    test_buy  = test_df["label"].mean() * 100
    log.info(f"Label distribution:")
    log.info(f"  Train buy rate: {train_buy:.1f}%")
    log.info(f"  Test  buy rate: {test_buy:.1f}%")

    # 10. Feature analysis
    log.info(f"\nFinal feature list ({len(feature_cols)} features):")
    for i, col in enumerate(feature_cols, 1):
        log.info(f"  {i:>2}. {col}")

    # 11. Save
    save_dataset(df_features_clean, feature_cols, train_df, test_df)

    return df_features_clean, feature_cols, train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="TradeQ Phase 3 — Dataset Builder"
    )
    parser.add_argument(
        "--test-days", type=int, default=DEFAULT_TEST_DAYS,
        help=f"Trading days to reserve for test set (default: {DEFAULT_TEST_DAYS})"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Print detailed feature statistics without building dataset"
    )
    args = parser.parse_args()

    log.info("TradeQ — Phase 3 Dataset Builder")
    log.info("─" * 55)

    if args.analyze:
        engine   = get_engine()
        df       = load_raw_features(engine)
        df_clean = drop_leakage_and_metadata(df.copy())
        df_clean["label"] = df["label"].values
        feature_cols = get_feature_columns(df_clean)
        df_clean = drop_sparse_columns(
            df_clean.drop(columns=["label"]), NULL_THRESHOLD
        )
        stats = analyze_features(df_clean, [c for c in feature_cols if c in df_clean.columns])
        print("\n" + stats.to_string(index=False))
        return

    df, feature_cols, train_df, test_df = build_dataset(args.test_days)

    # Final summary
    X_train, y_train = make_xy(train_df, feature_cols)
    X_test,  y_test  = make_xy(test_df,  feature_cols)

    print("\n" + "═" * 55)
    print("  DATASET BUILD COMPLETE")
    print("═" * 55)
    print(f"  Features          : {len(feature_cols)}")
    print(f"  Train rows        : {len(X_train):,}")
    print(f"  Test rows         : {len(X_test):,}")
    print(f"  Train buy rate    : {y_train.mean()*100:.1f}%")
    print(f"  Test  buy rate    : {y_test.mean()*100:.1f}%")
    print(f"  scale_pos_weight  : {(y_train==0).sum() / (y_train==1).sum():.2f}")
    print(f"  Saved to          : {CONFIG['paths']['models']}/dataset.parquet")
    print("═" * 55)
    print("\nNext step:")
    print("  python3 phase3/models/trainer.py")


if __name__ == "__main__":
    main()