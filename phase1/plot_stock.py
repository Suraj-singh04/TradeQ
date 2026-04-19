"""
TradeQ AI — Phase 1
File: phase1/plot_stock.py

What this script does:
  - Loads any stock CSV from data/raw/
  - Plots a professional candlestick chart with:
      * Green/red candles (up/down days)
      * Volume bars below the price chart
      * 20-day and 50-day Exponential Moving Averages (EMA)
      * 52-week high and low markers
      * Clean grid, labels, and title
  - Saves the chart as a PNG into data/charts/
  - Can be run for any stock by changing SYMBOL below

How to run:
  python3 phase1/plot_stock.py

  To plot a different stock, change the SYMBOL variable below,
  or pass it as an argument:
  python3 phase1/plot_stock.py TATAMOTORS

Author: TradeQ AI Project
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────
# Change SYMBOL to plot any stock you downloaded.
# Pass as command-line arg too:  python3 phase1/plot_stock.py TATAMOTORS
SYMBOL      = "RELIANCE"
DATA_DIR    = "data/raw"
CHARTS_DIR  = "data/charts"
EMA_SHORT   = 20    # Short-term EMA period (days)
EMA_LONG    = 50    # Long-term EMA period (days)

# Chart colors — easy to change if you want a different look
CLR_UP      = "#1D9E75"   # Green for up candles
CLR_DOWN    = "#D85A30"   # Red/coral for down candles
CLR_EMA20   = "#378ADD"   # Blue for 20-day EMA
CLR_EMA50   = "#EF9F27"   # Amber for 50-day EMA
CLR_52H     = "#1D9E75"   # Green dashed for 52-week high
CLR_52L     = "#D85A30"   # Red dashed for 52-week low
CLR_VOL_UP  = "#9FE1CB"   # Light green volume bars (up days)
CLR_VOL_DN  = "#F5C4B3"   # Light coral volume bars (down days)
CLR_BG      = "#FAFAF8"   # Chart background
CLR_GRID    = "#E8E6DF"   # Grid lines


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_stock_data(symbol: str, data_dir: str) -> pd.DataFrame:
    """
    Load a stock's CSV from data/raw/ and validate it.

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


# ─── Feature calculation ──────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.
    These same indicators will be used by the ML model in Phase 3.

    Adds:
        EMA_20:    20-day Exponential Moving Average
        EMA_50:    50-day Exponential Moving Average
        52W_High:  52-week rolling high
        52W_Low:   52-week rolling low
        Is_Up:     True if Close >= Open (green candle)
    """
    df = df.copy()
    df["EMA_20"]   = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean().round(2)
    df["EMA_50"]   = df["Close"].ewm(span=EMA_LONG,  adjust=False).mean().round(2)
    df["52W_High"] = df["High"].rolling(window=252).max()
    df["52W_Low"]  = df["Low"].rolling(window=252).min()
    df["Is_Up"]    = df["Close"] >= df["Open"]
    return df


# ─── Chart drawing ────────────────────────────────────────────────────────────

def draw_chart(df: pd.DataFrame, symbol: str) -> plt.Figure:
    """
    Draw a professional candlestick chart with volume, EMAs, and 52-week markers.

    Args:
        df:     DataFrame with OHLCV + indicators
        symbol: Stock name for the title

    Returns:
        A matplotlib Figure object.
    """
    fig = plt.figure(figsize=(16, 9), facecolor=CLR_BG)

    # Two panels: price (top, 70% height) and volume (bottom, 30%)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[7, 3], hspace=0.06)
    ax_price  = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1], sharex=ax_price)

    # Use numeric x-axis so candle widths are consistent
    x = range(len(df))
    dates = df.index

    # ── Candlesticks ──────────────────────────────────────────────────────────
    candle_width = 0.6
    wick_width   = 0.8

    for i, (idx, row) in enumerate(df.iterrows()):
        color  = CLR_UP if row["Is_Up"] else CLR_DOWN
        body_bottom = min(row["Open"], row["Close"])
        body_height = abs(row["Close"] - row["Open"])
        body_height = max(body_height, row["Close"] * 0.001)  # Tiny floor so doji candles are visible

        # Body (filled rectangle)
        ax_price.bar(
            i, body_height,
            bottom=body_bottom,
            width=candle_width,
            color=color,
            linewidth=0,
            zorder=3,
        )
        # Wick (thin vertical line from low to high)
        ax_price.plot(
            [i, i], [row["Low"], row["High"]],
            color=color,
            linewidth=wick_width,
            zorder=2,
        )

    # ── Moving averages ───────────────────────────────────────────────────────
    ax_price.plot(
        x, df["EMA_20"],
        color=CLR_EMA20, linewidth=1.4, label=f"EMA {EMA_SHORT}", zorder=4, alpha=0.9
    )
    ax_price.plot(
        x, df["EMA_50"],
        color=CLR_EMA50, linewidth=1.4, label=f"EMA {EMA_LONG}", zorder=4, alpha=0.9, linestyle="--"
    )

    # ── 52-week high / low lines ──────────────────────────────────────────────
    high_52w = df["High"].max()
    low_52w  = df["Low"].min()

    ax_price.axhline(
        high_52w, color=CLR_52H, linewidth=1, linestyle=":",
        label=f"52W High  ₹{high_52w:,.1f}", alpha=0.8, zorder=1
    )
    ax_price.axhline(
        low_52w, color=CLR_52L, linewidth=1, linestyle=":",
        label=f"52W Low   ₹{low_52w:,.1f}", alpha=0.8, zorder=1
    )

    # ── Volume bars ───────────────────────────────────────────────────────────
    vol_colors = [CLR_VOL_UP if up else CLR_VOL_DN for up in df["Is_Up"]]
    ax_volume.bar(x, df["Volume"] / 1_000_000, color=vol_colors, width=candle_width, linewidth=0)
    ax_volume.set_ylabel("Volume (M)", fontsize=10, color="#888780")

    # ── X-axis: show actual dates at sensible intervals ───────────────────────
    # Pick ~10 evenly spaced tick positions
    n = len(df)
    tick_step  = max(1, n // 10)
    tick_positions = list(range(0, n, tick_step))
    tick_labels    = [dates[i].strftime("%b '%y") for i in tick_positions]

    ax_price.set_xticks(tick_positions)
    ax_price.set_xticklabels([])             # Hide on price panel (shared x-axis)
    ax_volume.set_xticks(tick_positions)
    ax_volume.set_xticklabels(tick_labels, fontsize=9, color="#888780")

    # ── Styling ───────────────────────────────────────────────────────────────
    for ax in [ax_price, ax_volume]:
        ax.set_facecolor(CLR_BG)
        ax.grid(True, color=CLR_GRID, linewidth=0.5, linestyle="-", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(CLR_GRID)
        ax.spines["bottom"].set_color(CLR_GRID)
        ax.tick_params(colors="#888780")

    ax_price.set_ylabel("Price (INR ₹)", fontsize=10, color="#888780")
    ax_price.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f"₹{val:,.0f}")
    )

    # Current price annotation (last close)
    last_close = df["Close"].iloc[-1]
    ax_price.annotate(
        f" ₹{last_close:,.2f}",
        xy=(len(df) - 1, last_close),
        fontsize=10, fontweight="bold",
        color=CLR_UP if df["Is_Up"].iloc[-1] else CLR_DOWN,
        va="center",
    )

    # ── Title and legend ──────────────────────────────────────────────────────
    date_range = f"{dates[0].strftime('%d %b %Y')} — {dates[-1].strftime('%d %b %Y')}"
    change_pct = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
    sign       = "+" if change_pct >= 0 else ""

    fig.suptitle(
        f"{symbol}.NS   |   {date_range}   |   {sign}{change_pct:.1f}% over period",
        fontsize=13, fontweight="500", color="#2C2C2A",
        x=0.05, ha="left", y=0.97
    )

    ax_price.legend(
        loc="upper left", fontsize=9,
        framealpha=0.85, edgecolor=CLR_GRID,
        facecolor=CLR_BG,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(symbol: str = SYMBOL):
    symbol = symbol.upper().replace(".NS", "")
    log.info(f"TradeQ AI — Phase 1 Chart Generator")
    log.info(f"Stock: {symbol}")

    # Load and prepare data
    df = load_stock_data(symbol, DATA_DIR)
    df = add_indicators(df)

    # Print a quick stats summary to terminal
    last   = df.iloc[-1]
    prev   = df.iloc[-2]
    change = last["Close"] - prev["Close"]
    pct    = (change / prev["Close"]) * 100

    print("\n" + "─" * 50)
    print(f"  {symbol}.NS  —  Last trading day: {df.index[-1].date()}")
    print("─" * 50)
    print(f"  Open   : ₹{last['Open']:>10,.2f}")
    print(f"  High   : ₹{last['High']:>10,.2f}")
    print(f"  Low    : ₹{last['Low']:>10,.2f}")
    print(f"  Close  : ₹{last['Close']:>10,.2f}  ({'+' if change >= 0 else ''}{change:.2f}, {'+' if pct >= 0 else ''}{pct:.2f}%)")
    print(f"  Volume : {last['Volume']:>12,.0f} shares")
    print(f"  EMA 20 : ₹{last['EMA_20']:>10,.2f}")
    print(f"  EMA 50 : ₹{last['EMA_50']:>10,.2f}")
    print(f"  52W Hi : ₹{df['High'].max():>10,.2f}")
    print(f"  52W Lo : ₹{df['Low'].min():>10,.2f}")
    print("─" * 50 + "\n")

    # Draw and save chart
    os.makedirs(CHARTS_DIR, exist_ok=True)
    fig      = draw_chart(df, symbol)
    out_path = os.path.join(CHARTS_DIR, f"{symbol}_chart.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    log.info(f"Chart saved → {out_path}")

    # Also show it on screen
    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow passing symbol as command-line argument
    # Example: python3 phase1/plot_stock.py TATAMOTORS
    symbol = sys.argv[1] if len(sys.argv) > 1 else SYMBOL
    main(symbol)