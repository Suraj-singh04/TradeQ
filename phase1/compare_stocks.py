"""
TradeQ — Phase 1
File: phase1/compare_stocks.py

What this script does:
  - Loads multiple stocks from data/raw/ simultaneously
  - Compares them across 5 dimensions:
      * Cumulative return (who won over the period)
      * Daily volatility (who is the riskiest)
      * Sharpe ratio (best risk-adjusted return)
      * Correlation matrix (which stocks move together)
      * Rolling 30-day return race (how rankings shifted over time)
  - Prints a ranked comparison table to the terminal
  - Saves a 5-panel comparison chart to data/charts/
  - Identifies which stocks are best candidates for Phase 3 modeling

Why this matters for TradeQ:
  Your prediction model in Phase 3 can't treat all 50 stocks equally.
  Some are too correlated (predicting one predicts the other — redundant).
  Some are too volatile (hard to predict reliably).
  Some have the best risk-adjusted returns (best ROI for the model's effort).
  This script tells you exactly which stocks to prioritize.

How to run:
  python3 phase1/compare_stocks.py

  Edit SYMBOLS below to compare any set of stocks you downloaded.

Author: TradeQ Project
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
# Edit this list to compare any stocks you downloaded.
# Keep it between 5 and 15 for readable charts.
from config import CONFIG, get_symbols

SYMBOLS    = get_symbols("nifty50")
DATA_DIR   = CONFIG["paths"]["raw_data"]
CHARTS_DIR = CONFIG["paths"]["charts"]
PALETTE    = CONFIG["charts"]["palette"]

CLR_BG   = CONFIG["charts"]["background"]
CLR_GRID = CONFIG["charts"]["grid_color"]
CLR_TEXT = CONFIG["charts"]["text_primary"]
CLR_MUTED= CONFIG["charts"]["text_muted"]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_all_stocks(symbols: list[str], data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load CSVs for all symbols using central load_stock_data.
    Skips any that are missing with a warning.

    Returns:
        Dict mapping symbol name → DataFrame
    """
    stocks = {}
    for sym in symbols:
        try:
            df = load_stock_data(sym, data_dir)
            stocks[sym] = df
        except FileNotFoundError:
            log.warning(f"  {sym:<15} — CSV not found, skipping. Run download_stocks.py first.")
        except Exception as e:
            log.error(f"  {sym:<15} — Failed to load: {e}")
    return stocks


def align_to_common_dates(stocks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Trim all stocks to share the same date range.
    Without this, gaps in one stock cause NaN issues in correlation and charts.
    """
    # Find the date range that all stocks share
    start = max(df.index[0]  for df in stocks.values())
    end   = min(df.index[-1] for df in stocks.values())
    aligned = {sym: df.loc[start:end] for sym, df in stocks.items()}
    log.info(f"Aligned to common window: {start.date()} → {end.date()}")
    return aligned


# ─── Metrics computation ──────────────────────────────────────────────────────

def compute_all_metrics(stocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute one summary row per stock.
    Returns a DataFrame with all comparison metrics — the ranking table.
    """
    rows = []
    for sym, df in stocks.items():
        close   = df["Close"]
        returns = close.pct_change().dropna() * 100

        total_ret  = (close.iloc[-1] / close.iloc[0] - 1) * 100
        ann_ret    = returns.mean() * 252
        ann_vol    = returns.std() * np.sqrt(252)
        sharpe     = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        win_rate   = (returns > 0).mean() * 100
        max_dd     = _max_drawdown(close)
        avg_vol    = df["Volume"].mean()

        rows.append({
            "Symbol":       sym,
            "Total Ret %":  round(total_ret,  2),
            "Ann Ret %":    round(ann_ret,     2),
            "Ann Vol %":    round(ann_vol,     2),
            "Sharpe":       round(sharpe,      3),
            "Win Rate %":   round(win_rate,    1),
            "Max DD %":     round(max_dd,      2),
            "Avg Volume":   int(avg_vol),
        })

    metrics = pd.DataFrame(rows).set_index("Symbol")
    return metrics


def build_returns_matrix(stocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a DataFrame of daily % returns with one column per stock.
    Used for correlation and cumulative return calculations.
    """
    ret_dict = {}
    for sym, df in stocks.items():
        ret_dict[sym] = df["Close"].pct_change() * 100
    return pd.DataFrame(ret_dict).dropna()


def _max_drawdown(prices: pd.Series) -> float:
    rolling_max = prices.cummax()
    drawdown    = (prices - rolling_max) / rolling_max * 100
    return round(drawdown.min(), 2)


# ─── Terminal output ──────────────────────────────────────────────────────────

def print_comparison(metrics: pd.DataFrame, corr: pd.DataFrame) -> None:
    """Print the ranked comparison table and correlation insights."""

    print(f"\n{'═' * 72}")
    print(f"  TRADEQ — STOCK COMPARISON REPORT")
    print(f"{'═' * 72}")

    # Rank by Sharpe (best risk-adjusted return)
    ranked = metrics.sort_values("Sharpe", ascending=False)

    print(f"\n  RANKED BY SHARPE RATIO  (higher = better risk-adjusted return)")
    print(f"  {'Symbol':<14} {'Total%':>8} {'AnnRet%':>8} {'AnnVol%':>8} "
          f"{'Sharpe':>7} {'WinRate%':>9} {'MaxDD%':>8}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*9} {'─'*8}")

    for i, (sym, row) in enumerate(ranked.iterrows(), 1):
        medal = " #1" if i == 1 else "   "
        ret_sign = "+" if row["Total Ret %"] >= 0 else ""
        ann_sign = "+" if row["Ann Ret %"]   >= 0 else ""
        print(
            f"{medal} {sym:<14} "
            f"{ret_sign}{row['Total Ret %']:>7.2f}% "
            f"{ann_sign}{row['Ann Ret %']:>7.2f}% "
            f"{row['Ann Vol %']:>8.2f}% "
            f"{row['Sharpe']:>7.3f} "
            f"{row['Win Rate %']:>8.1f}% "
            f"{row['Max DD %']:>8.2f}%"
        )

    # Correlation insights
    print(f"\n  CORRELATION INSIGHTS  (1.0 = move in lockstep, 0 = independent)")
    print(f"  {'─' * 60}")

    # Find highest and lowest correlated pairs
    pairs = []
    syms  = corr.columns.tolist()
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            pairs.append((syms[i], syms[j], corr.iloc[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Most correlated pairs (avoid using both — they tell the same story):")
    for s1, s2, c in pairs[:3]:
        print(f"    {s1} ↔ {s2:<14}  corr = {c:.3f}")

    print(f"\n  Least correlated pairs (good for diversification & independent signals):")
    for s1, s2, c in pairs[-3:]:
        print(f"    {s1} ↔ {s2:<14}  corr = {c:.3f}")

    # Best Phase 3 candidates
    print(f"\n  BEST CANDIDATES FOR PHASE 3 MODELING")
    print(f"  {'─' * 60}")
    print(f"  Criteria: Sharpe > 0.5, Volatility < 30%, Win Rate > 48%")
    candidates = metrics[
        (metrics["Sharpe"]     > 0.5) &
        (metrics["Ann Vol %"]  < 30)  &
        (metrics["Win Rate %"] > 48)
    ].sort_values("Sharpe", ascending=False)

    if len(candidates) > 0:
        for sym in candidates.index:
            print(f"    {sym:<15}  Sharpe {candidates.loc[sym,'Sharpe']:.3f}  "
                  f"Vol {candidates.loc[sym,'Ann Vol %']:.1f}%  "
                  f"Win {candidates.loc[sym,'Win Rate %']:.1f}%")
    else:
        print("    No stocks met all 3 criteria — consider relaxing thresholds.")

    print(f"\n{'═' * 72}\n")


# ─── Chart ────────────────────────────────────────────────────────────────────

def draw_comparison_chart(
    stocks:  dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    metrics: pd.DataFrame,
    corr:    pd.DataFrame,
) -> plt.Figure:
    """
    Draw a 5-panel comparison chart:
      1. Cumulative return race (top, full width)
      2. Annualized return vs volatility scatter
      3. Sharpe ratio bar chart
      4. Correlation heatmap
      5. Rolling 30-day return for top 5
    """
    fig = plt.figure(figsize=(20, 14), facecolor=CLR_BG)
    fig.suptitle(
        f"TradeQ — Multi-Stock Comparison   |   {len(stocks)} stocks",
        fontsize=14, fontweight="500", color=CLR_TEXT, x=0.02, ha="left", y=0.99
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    color_map = {sym: PALETTE[i % len(PALETTE)] for i, sym in enumerate(stocks.keys())}

    # ── Panel 1: Cumulative return race (full top row) ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for sym, df in stocks.items():
        cum = (df["Close"] / df["Close"].iloc[0] - 1) * 100
        ax1.plot(df.index, cum, color=color_map[sym], linewidth=1.4,
                 label=sym, alpha=0.9)
    ax1.axhline(0, color=CLR_MUTED, linewidth=0.8, linestyle="--")
    ax1.set_title("Cumulative return race (%)", fontsize=11, color=CLR_TEXT, loc="left")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax1.legend(
        loc="upper left", fontsize=8, ncol=min(len(stocks), 5),
        framealpha=0.85, edgecolor=CLR_GRID, facecolor=CLR_BG
    )
    _style_ax(ax1)

    # ── Panel 2: Return vs Volatility scatter (risk/reward map) ───────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for sym in metrics.index:
        x = metrics.loc[sym, "Ann Vol %"]
        y = metrics.loc[sym, "Ann Ret %"]
        ax2.scatter(x, y, color=color_map[sym], s=80, zorder=3)
        ax2.annotate(
            sym, (x, y),
            fontsize=7, color=color_map[sym],
            xytext=(4, 3), textcoords="offset points"
        )
    ax2.axhline(0, color=CLR_MUTED, linewidth=0.8, linestyle="--")
    ax2.set_title("Return vs Volatility  (top-right = best)", fontsize=11, color=CLR_TEXT, loc="left")
    ax2.set_xlabel("Annualized volatility (%)", fontsize=9, color=CLR_MUTED)
    ax2.set_ylabel("Annualized return (%)", fontsize=9, color=CLR_MUTED)
    _style_ax(ax2)

    # ── Panel 3: Sharpe ratio bar chart ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ranked_sharpe = metrics["Sharpe"].sort_values(ascending=True)
    bar_colors    = [color_map[s] for s in ranked_sharpe.index]
    bars = ax3.barh(ranked_sharpe.index, ranked_sharpe.values,
                    color=bar_colors, alpha=0.85, height=0.6)
    ax3.axvline(0, color=CLR_MUTED, linewidth=0.8)
    ax3.axvline(1.0, color=CLR_GRID, linewidth=1, linestyle="--",
                label="Sharpe = 1.0 (good)")
    # Value labels on bars
    for bar, val in zip(bars, ranked_sharpe.values):
        ax3.text(
            val + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=8, color=CLR_TEXT
        )
    ax3.set_title("Sharpe ratio (higher = better)", fontsize=11, color=CLR_TEXT, loc="left")
    ax3.legend(fontsize=8, framealpha=0)
    _style_ax(ax3)

    # ── Panel 4: Correlation heatmap ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    n   = len(corr)
    im  = ax4.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax4.set_xticks(range(n))
    ax4.set_yticks(range(n))
    ax4.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax4.set_yticklabels(corr.index, fontsize=7)

    # Annotate each cell with correlation value
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            txt_color = "white" if abs(val) > 0.7 else CLR_TEXT
            ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6, color=txt_color)

    ax4.set_title("Correlation matrix", fontsize=11, color=CLR_TEXT, loc="left")
    plt.colorbar(im, ax=ax4, fraction=0.04, pad=0.02)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax4.spines["bottom"].set_visible(False)
    ax4.set_facecolor(CLR_BG)
    ax4.tick_params(colors=CLR_MUTED)

    # ── Panel 5: Rolling 30-day return — top 5 by Sharpe ──────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    top5 = metrics["Sharpe"].sort_values(ascending=False).head(5).index.tolist()
    for sym in top5:
        df       = stocks[sym]
        roll_ret = (df["Close"].pct_change() * 100).rolling(30).mean()
        ax5.plot(df.index, roll_ret, color=color_map[sym],
                 linewidth=1.3, label=sym, alpha=0.9)
    ax5.axhline(0, color=CLR_MUTED, linewidth=0.8, linestyle="--")
    ax5.set_title(
        "Rolling 30-day avg return — top 5 by Sharpe  "
        "(positive = trending up, negative = trending down)",
        fontsize=11, color=CLR_TEXT, loc="left"
    )
    ax5.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}%"))
    ax5.legend(loc="upper left", fontsize=9, framealpha=0.85,
               edgecolor=CLR_GRID, facecolor=CLR_BG)
    _style_ax(ax5)

    return fig


def _style_ax(ax) -> None:
    ax.set_facecolor(CLR_BG)
    ax.grid(True, color=CLR_GRID, linewidth=0.5, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CLR_GRID)
    ax.spines["bottom"].set_color(CLR_GRID)
    ax.tick_params(colors=CLR_MUTED, labelsize=8)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("TradeQ — Phase 1 Multi-Stock Comparison")
    log.info(f"Comparing: {', '.join(SYMBOLS)}")
    log.info("─" * 50)

    # Load and align all stocks
    stocks  = load_all_stocks(SYMBOLS, DATA_DIR)
    if len(stocks) < 2:
        log.error("Need at least 2 stocks loaded. Check data/raw/ folder.")
        sys.exit(1)

    stocks  = align_to_common_dates(stocks)

    # Compute metrics and correlation
    metrics = compute_all_metrics(stocks)
    returns = build_returns_matrix(stocks)
    corr    = returns.corr().round(3)

    # Print and chart
    print_comparison(metrics, corr)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    fig      = draw_comparison_chart(stocks, returns, metrics, corr)
    out_path = os.path.join(CHARTS_DIR, "comparison_chart.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    log.info(f"Comparison chart saved → {out_path}")

    # Also save metrics as CSV — useful reference for Phase 3 stock selection
    metrics_path = os.path.join(CHARTS_DIR, "stock_metrics.csv")
    metrics.sort_values("Sharpe", ascending=False).to_csv(metrics_path)
    log.info(f"Metrics CSV saved → {metrics_path}  (open this in any spreadsheet)")

    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()