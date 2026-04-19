import os
import logging
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)

def load_stock_data(symbol: str, data_dir: str) -> pd.DataFrame:
    """
    Load a stock's CSV from data_dir and validate it.

    Args:
        symbol:   Stock name without .NS e.g. "RELIANCE"
        data_dir: Folder containing CSV files

    Returns:
        Cleaned DataFrame with OHLCV columns.
    """
    filepath = os.path.join(data_dir, f"{symbol}.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No CSV found for {symbol} at {filepath}\n"
            f"Run download_stocks.py first to fetch the data."
        )

    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Sort chronologically — always do this, CSVs can sometimes be unordered
    df = df.sort_index()

    log.info(f"Loaded {symbol}: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df
