"""
TradeQ AI — Phase 1
File: phase1/explore_stock.py

What this script does:
  - Loads any stock CSV from data/raw/
  - Runs a full statistical deep-dive — the kind of analysis a quant does
    before building any model
  - Covers: returns, volatility, streaks, best/worst days, volume analysis,
    monthly performance, and correlation between volume and price moves
  - Saves a multi-panel analysis chart to data/charts/
  - Everything computed here directly informs which features we build in Phase 2

Why this matters:
  Before you teach an AI to predict a stock, you need to understand how that
  stock actually behaves. Does it trend or mean-revert? Are big volume days
  followed by continuation or reversal? What month is historically strongest?
  This script answers all of that.

How to run:
  python3 phase1/explore_stock.py
  python3 phase1/explore_stock.py HDFCBANK

Author: TradeQ AI Project
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# ─── Add root to path for utils and config ────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_stock_data

warnings.filterwarnings("ignore")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, get_raw_path

SYMBOL     = "RELIANCE"
DATA_DIR   = CONFIG["paths"]["raw_data"]
CHARTS_DIR = CONFIG["paths"]["charts"]

# Chart colors
CLR_POS    = CONFIG["charts"]["candle_up"]
CLR_NEG    = CONFIG["charts"]["candle_down"]
CLR_BLUE   = CONFIG["charts"]["ema_short"]
CLR_AMBER  = CONFIG["charts"]["ema_long"]
CLR_PURPLE = CONFIG["charts"]["palette"][4]
CLR_BG     = CONFIG["charts"]["background"]
CLR_GRID   = CONFIG["charts"]["grid_color"]
CLR_TEXT   = CONFIG["charts"]["text_primary"]
CLR_MUTED  = CONFIG["charts"]["text_muted"]


# ─── Data loading ─────────────────────────────────────────────────────────────

# ─── Feature engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived columns needed for analysis.
    These are early versions of the Phase 2 features — same logic, built manually.
    """
    df = df.copy()

    # Daily returns (percentage change in close price)
    df["Daily_Return"]     = df["Close"].pct_change() * 100

    # Log returns — more statistically well-behaved than simple returns
    df["Log_Return"]       = np.log(df["Close"] / df["Close"].shift(1)) * 100

    # True range — captures gap days properly
    df["Prev_Close"]       = df["Close"].shift(1)
    df["True_Range"]       = (
        pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Prev_Close"]).abs(),
            (df["Low"]  - df["Prev_Close"]).abs(),
        ], axis=1).max(axis=1)
    )

    # Average True Range (14-day) — volatility measure
    df["ATR_14"]           = df["True_Range"].rolling(14).mean().round(2)

    # Volume spike: today's volume vs 20-day average
    df["Vol_MA20"]         = df["Volume"].rolling(20).mean()
    df["Volume_Spike"]     = (df["Volume"] / df["Vol_MA20"]).round(2)

    # Is this candle green (up) or red (down)?
    df["Is_Up"]            = df["Close"] >= df["Open"]

    # Body size as % of price — big body = strong conviction day
    df["Body_Pct"]         = ((df["Close"] - df["Open"]).abs() / df["Open"] * 100).round(3)

    # Cumulative return from start of dataset
    df["Cumulative_Return"] = ((df["Close"] / df["Close"].iloc[0]) - 1) * 100

    # Rolling 20-day volatility (annualized)
    df["Volatility_20d"]   = df["Daily_Return"].rolling(20).std() * np.sqrt(252)

    # Month and weekday for calendar analysis
    df["Month"]            = df.index.month
    df["Weekday"]          = df.index.dayofweek   # 0=Mon, 4=Fri
    df["Month_Name"]       = df.index.strftime("%b")
    df["Weekday_Name"]     = df.index.strftime("%a")

    # Next-day return — what actually happened after each day
    # This is exactly the label we'll predict in Phase 3
    df["Next_Day_Return"]  = df["Daily_Return"].shift(-1)

    return df


# ─── Statistical analysis ─────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame, symbol: str) -> dict:
    """
    Compute every statistic we care about. Returns a dict of results.
    This is the data that will be printed and plotted.
    """
    r = df["Daily_Return"].dropna()

    # ── Basic return stats ────────────────────────────────────────────────────
    stats = {
        "symbol":           symbol,
        "total_days":       len(df),
        "date_from":        df.index[0].date(),
        "date_to":          df.index[-1].date(),
        "start_price":      df["Close"].iloc[0],
        "end_price":        df["Close"].iloc[-1],
        "total_return_pct": df["Cumulative_Return"].iloc[-1],
        "mean_daily_ret":   r.mean(),
        "median_daily_ret": r.median(),
        "std_daily_ret":    r.std(),
        "annualized_vol":   r.std() * np.sqrt(252),
        "annualized_ret":   r.mean() * 252,
        "sharpe_ratio":     (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0,

        # ── Best and worst days ───────────────────────────────────────────────
        "best_day_ret":     r.max(),
        "best_day_date":    r.idxmax().date(),
        "worst_day_ret":    r.min(),
        "worst_day_date":   r.idxmin().date(),

        # ── Win rate ──────────────────────────────────────────────────────────
        "up_days":          int((r > 0).sum()),
        "down_days":        int((r < 0).sum()),
        "flat_days":        int((r == 0).sum()),
        "win_rate":         (r > 0).mean() * 100,

        # ── Volatility ────────────────────────────────────────────────────────
        "max_drawdown":     _max_drawdown(df["Close"]),
        "avg_true_range":   df["ATR_14"].mean(),

        # ── Volume ────────────────────────────────────────────────────────────
        "avg_volume":       df["Volume"].mean(),
        "max_volume_date":  df["Volume"].idxmax().date(),
        "max_volume":       df["Volume"].max(),

        # ── Volume-return correlation (key insight for Phase 2) ───────────────
        "vol_spike_corr":   df[["Volume_Spike", "Daily_Return"]].dropna().corr().iloc[0, 1],
        "vol_next_corr":    df[["Volume_Spike", "Next_Day_Return"]].dropna().corr().iloc[0, 1],
    }

    # ── Consecutive streak analysis ───────────────────────────────────────────
    stats["max_up_streak"]   = _max_streak(df["Is_Up"], True)
    stats["max_down_streak"] = _max_streak(df["Is_Up"], False)

    # ── Monthly average return ────────────────────────────────────────────────
    stats["monthly_avg"] = (
        df.groupby("Month_Name")["Daily_Return"]
        .mean()
        .reindex(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        .dropna()
    )

    # ── Weekday average return ────────────────────────────────────────────────
    stats["weekday_avg"] = (
        df.groupby("Weekday_Name")["Daily_Return"]
        .mean()
        .reindex(["Mon","Tue","Wed","Thu","Fri"])
        .dropna()
    )

    # ── Big move days (> 2% move in either direction) ─────────────────────────
    big_moves = df[df["Daily_Return"].abs() > 2.0].copy()
    stats["big_move_days"]      = len(big_moves)
    stats["big_move_pct"]       = len(big_moves) / len(df) * 100
    # On big volume days (spike > 1.5x), what was the average return?
    stats["high_vol_avg_return"] = df[df["Volume_Spike"] > 1.5]["Daily_Return"].mean()
    stats["low_vol_avg_return"]  = df[df["Volume_Spike"] <= 1.5]["Daily_Return"].mean()

    return stats


def _max_drawdown(prices: pd.Series) -> float:
    """Calculate the maximum drawdown percentage from peak to trough."""
    rolling_max = prices.cummax()
    drawdown    = (prices - rolling_max) / rolling_max * 100
    return drawdown.min()


def _max_streak(is_up: pd.Series, direction: bool) -> int:
    """Find the longest consecutive streak of up or down days."""
    max_streak = current = 0
    for val in is_up:
        if val == direction:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


# ─── Terminal output ──────────────────────────────────────────────────────────

def print_analysis(s: dict) -> None:
    """Print the full analysis in a readable format to the terminal."""
    sep = "─" * 58

    print(f"\n{'═' * 58}")
    print(f"  TRADEQ — STOCK EXPLORER")
    print(f"  {s['symbol']}.NS   |   {s['date_from']} → {s['date_to']}")
    print(f"{'═' * 58}")

    print(f"\n  PRICE SUMMARY")
    print(sep)
    print(f"  Start price     : ₹{s['start_price']:>10,.2f}")
    print(f"  End price       : ₹{s['end_price']:>10,.2f}")
    sign = "+" if s["total_return_pct"] >= 0 else ""
    print(f"  Total return    : {sign}{s['total_return_pct']:.2f}%  over {s['total_days']} trading days")

    print(f"\n  DAILY RETURNS")
    print(sep)
    print(f"  Mean daily ret  : {s['mean_daily_ret']:+.4f}%")
    print(f"  Median daily ret: {s['median_daily_ret']:+.4f}%")
    print(f"  Std deviation   : {s['std_daily_ret']:.4f}%")
    print(f"  Annualized ret  : {s['annualized_ret']:+.2f}%")
    print(f"  Annualized vol  : {s['annualized_vol']:.2f}%")
    print(f"  Sharpe ratio    : {s['sharpe_ratio']:.3f}  (> 1.0 is good, > 2.0 is excellent)")

    print(f"\n  WIN RATE")
    print(sep)
    print(f"  Up days         : {s['up_days']}  ({s['win_rate']:.1f}%)")
    print(f"  Down days       : {s['down_days']}  ({100 - s['win_rate'] - (s['flat_days']/s['total_days']*100):.1f}%)")
    print(f"  Flat days       : {s['flat_days']}")
    print(f"  Max up streak   : {s['max_up_streak']} consecutive green days")
    print(f"  Max down streak : {s['max_down_streak']} consecutive red days")

    print(f"\n  BEST & WORST DAYS")
    print(sep)
    print(f"  Best day        : {s['best_day_ret']:+.2f}%  on {s['best_day_date']}")
    print(f"  Worst day       : {s['worst_day_ret']:+.2f}%  on {s['worst_day_date']}")
    print(f"  Max drawdown    : {s['max_drawdown']:.2f}%  (peak-to-trough)")
    print(f"  Big move days   : {s['big_move_days']} days ({s['big_move_pct']:.1f}%) had >2% swing")

    print(f"\n  VOLUME ANALYSIS  ← key Phase 2 insight")
    print(sep)
    print(f"  Avg daily volume: {s['avg_volume']:>15,.0f} shares")
    print(f"  Highest vol day : {s['max_volume_date']}  ({s['max_volume']:,.0f} shares)")
    print(f"  High-vol day avg return : {s['high_vol_avg_return']:+.4f}%  (volume spike > 1.5x)")
    print(f"  Low-vol  day avg return : {s['low_vol_avg_return']:+.4f}%  (volume spike ≤ 1.5x)")
    print(f"  Volume ↔ same-day return corr  : {s['vol_spike_corr']:+.4f}")
    print(f"  Volume ↔ next-day return corr  : {s['vol_next_corr']:+.4f}")
    print(f"  (A positive next-day corr means big volume days tend to continue)")

    print(f"\n  MONTHLY AVERAGE RETURN")
    print(sep)
    for month, ret in s["monthly_avg"].items():
        bar = "█" * int(abs(ret) * 8)
        sign = "+" if ret >= 0 else ""
        clr  = ""
        print(f"  {month:<4}  {sign}{ret:.3f}%  {bar}")

    print(f"\n  WEEKDAY AVERAGE RETURN")
    print(sep)
    for day, ret in s["weekday_avg"].items():
        bar = "█" * int(abs(ret) * 8)
        sign = "+" if ret >= 0 else ""
        print(f"  {day}  {sign}{ret:.3f}%  {bar}")

    print(f"\n{'═' * 58}\n")


# ─── Chart ────────────────────────────────────────────────────────────────────

def draw_analysis_chart(df: pd.DataFrame, s: dict) -> plt.Figure:
    """
    Draw a 6-panel analysis chart covering returns, distribution,
    drawdown, volume, monthly performance, and cumulative return.
    """
    fig = plt.figure(figsize=(18, 12), facecolor=CLR_BG)
    fig.suptitle(
        f"{s['symbol']}.NS — Statistical Analysis   |   "
        f"{s['date_from']} → {s['date_to']}   |   "
        f"Sharpe: {s['sharpe_ratio']:.2f}   Win rate: {s['win_rate']:.1f}%",
        fontsize=13, fontweight="500", color=CLR_TEXT, x=0.02, ha="left", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Cumulative return ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    cum = df["Cumulative_Return"].dropna()
    colors_line = [CLR_POS if v >= 0 else CLR_NEG for v in cum]
    ax1.plot(df.index[len(df) - len(cum):], cum, color=CLR_BLUE, linewidth=1.5)
    ax1.fill_between(
        df.index[len(df) - len(cum):], cum, 0,
        where=(cum >= 0), alpha=0.12, color=CLR_POS
    )
    ax1.fill_between(
        df.index[len(df) - len(cum):], cum, 0,
        where=(cum < 0), alpha=0.12, color=CLR_NEG
    )
    ax1.axhline(0, color=CLR_MUTED, linewidth=0.8, linestyle="--")
    ax1.set_title("Cumulative return (%)", fontsize=11, color=CLR_TEXT, loc="left")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    _style_ax(ax1)

    # ── Panel 2: Rolling 20-day volatility ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    vol = df["Volatility_20d"].dropna()
    ax2.plot(df.index[len(df) - len(vol):], vol, color=CLR_AMBER, linewidth=1.2)
    ax2.fill_between(df.index[len(df) - len(vol):], vol, alpha=0.15, color=CLR_AMBER)
    ax2.set_title("Rolling volatility (20d ann.)", fontsize=11, color=CLR_TEXT, loc="left")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _style_ax(ax2)

    # ── Panel 3: Daily return distribution histogram ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    returns = df["Daily_Return"].dropna()
    n_bins  = 50
    counts, bin_edges = np.histogram(returns, bins=n_bins)
    for i in range(len(counts)):
        mid   = (bin_edges[i] + bin_edges[i+1]) / 2
        color = CLR_POS if mid >= 0 else CLR_NEG
        ax3.bar(mid, counts[i], width=(bin_edges[1]-bin_edges[0])*0.9,
                color=color, alpha=0.75, linewidth=0)
    ax3.axvline(returns.mean(), color=CLR_BLUE, linewidth=1.5,
                linestyle="--", label=f"Mean {returns.mean():+.3f}%")
    ax3.set_title("Daily return distribution", fontsize=11, color=CLR_TEXT, loc="left")
    ax3.set_xlabel("Return (%)", fontsize=9, color=CLR_MUTED)
    ax3.legend(fontsize=8, framealpha=0)
    _style_ax(ax3)

    # ── Panel 4: Volume spike vs next-day return scatter ──────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    scatter_df = df[["Volume_Spike", "Next_Day_Return"]].dropna()
    # Cap display to ±5% return and ≤ 4x volume spike for readability
    scatter_df = scatter_df[
        (scatter_df["Next_Day_Return"].abs() <= 5) &
        (scatter_df["Volume_Spike"] <= 4)
    ]
    colors_sc = [CLR_POS if v >= 0 else CLR_NEG for v in scatter_df["Next_Day_Return"]]
    ax4.scatter(
        scatter_df["Volume_Spike"], scatter_df["Next_Day_Return"],
        c=colors_sc, alpha=0.4, s=12, linewidths=0
    )
    ax4.axhline(0, color=CLR_MUTED, linewidth=0.8, linestyle="--")
    ax4.axvline(1.0, color=CLR_MUTED, linewidth=0.8, linestyle=":")
    # Trend line
    if len(scatter_df) > 10:
        z = np.polyfit(scatter_df["Volume_Spike"], scatter_df["Next_Day_Return"], 1)
        p = np.poly1d(z)
        xs = np.linspace(scatter_df["Volume_Spike"].min(), scatter_df["Volume_Spike"].max(), 100)
        ax4.plot(xs, p(xs), color=CLR_PURPLE, linewidth=1.5,
                 label=f"Corr: {s['vol_next_corr']:+.3f}")
    ax4.set_title("Volume spike → next-day return", fontsize=11, color=CLR_TEXT, loc="left")
    ax4.set_xlabel("Volume spike (x avg)", fontsize=9, color=CLR_MUTED)
    ax4.set_ylabel("Next-day return (%)", fontsize=9, color=CLR_MUTED)
    ax4.legend(fontsize=8, framealpha=0)
    _style_ax(ax4)

    # ── Panel 5: Monthly average return bar chart ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    months = s["monthly_avg"]
    bar_colors = [CLR_POS if v >= 0 else CLR_NEG for v in months.values]
    ax5.bar(months.index, months.values, color=bar_colors, alpha=0.8, width=0.6)
    ax5.axhline(0, color=CLR_MUTED, linewidth=0.8)
    ax5.set_title("Avg return by month", fontsize=11, color=CLR_TEXT, loc="left")
    ax5.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}%"))
    ax5.tick_params(axis="x", labelsize=8, rotation=45)
    _style_ax(ax5)

    # ── Panel 6: Drawdown over time ────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    prices      = df["Close"]
    rolling_max = prices.cummax()
    drawdown    = (prices - rolling_max) / rolling_max * 100
    ax6.fill_between(df.index, drawdown, 0, color=CLR_NEG, alpha=0.5)
    ax6.plot(df.index, drawdown, color=CLR_NEG, linewidth=0.8)
    ax6.set_title(
        f"Drawdown from peak  |  Max: {s['max_drawdown']:.2f}%",
        fontsize=11, color=CLR_TEXT, loc="left"
    )
    ax6.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _style_ax(ax6)

    # ── Panel 7: Weekday average return ───────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    days       = s["weekday_avg"]
    day_colors = [CLR_POS if v >= 0 else CLR_NEG for v in days.values]
    ax7.bar(days.index, days.values, color=day_colors, alpha=0.8, width=0.5)
    ax7.axhline(0, color=CLR_MUTED, linewidth=0.8)
    ax7.set_title("Avg return by weekday", fontsize=11, color=CLR_TEXT, loc="left")
    ax7.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}%"))
    _style_ax(ax7)

    return fig


def _style_ax(ax) -> None:
    """Apply consistent styling to every chart panel."""
    ax.set_facecolor(CLR_BG)
    ax.grid(True, color=CLR_GRID, linewidth=0.5, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CLR_GRID)
    ax.spines["bottom"].set_color(CLR_GRID)
    ax.tick_params(colors=CLR_MUTED, labelsize=8)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(symbol: str = SYMBOL):
    symbol = symbol.upper().replace(".NS", "")
    log.info(f"TradeQ — Phase 1 Explorer")
    log.info(f"Stock: {symbol}")

    df = load_stock_data(symbol, DATA_DIR)
    df = engineer_features(df)

    stats = compute_stats(df, symbol)
    print_analysis(stats)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    fig      = draw_analysis_chart(df, stats)
    out_path = os.path.join(CHARTS_DIR, f"{symbol}_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    log.info(f"Analysis chart saved → {out_path}")

    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else SYMBOL
    main(symbol)