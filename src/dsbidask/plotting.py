"""Plotting helpers (English labels, no font dependencies). Default resolution is ``DPI`` (150)."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from .data import Quotes
from .pricing import european_call_bid_ask, european_put_bid_ask

BID, ASK = "#32CD32", "#DC143C"
LINES = ["#4B8BBE", "#FFB31A", "#32CD32", "#DC143C"]
DPI = 150


def _finish(fig: Figure, path, dpi: Optional[int] = None) -> Figure:
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi or DPI)
    return fig


def plot_market_quotes(K: ArrayLike, bid: ArrayLike, ask: ArrayLike, title: str, path=None) -> Figure:
    """Market bid/ask prices and spreads against strike (thesis Fig. 4 and 5)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.scatter(K, ask, marker=".", color=ASK, label="Ask")
    a1.scatter(K, bid, marker=".", color=BID, label="Bid")
    a1.set(xlabel="Strike", ylabel="Price", title=f"{title}: bid and ask"); a1.legend()
    a2.scatter(K, np.asarray(ask) - np.asarray(bid), marker="+", color=ASK)
    a2.set(xlabel="Strike", ylabel="Ask - bid", title=f"{title}: spread")
    return _finish(fig, path)


def plot_spread_vs_epsilon(curves: Mapping[str, Tuple[NDArray, NDArray]], ylabel: str, title: str, path=None) -> Figure:
    """Spread curves vs epsilon (thesis Fig. 1); ``curves`` maps a label to ``(eps, spread)``."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, (lab, (e, s)) in zip(LINES, curves.items()):
        ax.plot(e, s, color=c, label=lab)
    ax.set(xlabel=r"$\epsilon$", ylabel=ylabel, title=title); ax.legend()
    return _finish(fig, path)


def plot_calibration_fit(q: Quotes, params: Mapping[str, float], R: float, T: int, path=None) -> Figure:
    """Model bid/ask curves vs market quotes for calls and puts (thesis Fig. 8)."""
    u, d, b_d = params["u"], params["d"], params["b_d"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, (kind, K, mb, ma, fn) in enumerate([
        ("Call", q.K_call, q.C_bid, q.C_ask, european_call_bid_ask),
        ("Put", q.K_put, q.P_bid, q.P_ask, european_put_bid_ask)]):
        grid = np.linspace(max(K.min(), 1e-6), K.max(), 200)
        mod = np.array([fn(q.S0_bid, k, u, d, b_d, R, T) for k in grid])
        top, bot = axes[0, col], axes[1, col]
        top.plot(grid, mod[:, 1], color=ASK, label="Model ask"); top.plot(grid, mod[:, 0], color=BID, label="Model bid")
        top.scatter(K, ma, marker=".", color=ASK); top.scatter(K, mb, marker=".", color=BID)
        top.set(xlabel="Strike", ylabel="Price", title=f"{kind}: model vs market"); top.legend()
        bot.plot(grid, mod[:, 1] - mod[:, 0], color=ASK, label="Model spread")
        bot.scatter(K, ma - mb, marker="+", color="k", label="Market spread")
        bot.set(xlabel="Strike", ylabel="Ask - bid", title=f"{kind}: spread"); bot.legend()
    return _finish(fig, path)


def plot_tree(values: NDArray, title: str = "", path=None, fmt: str = "{:.2f}", ax=None, annotate: bool = True,
              highlight: Optional[NDArray] = None) -> Optional[Figure]:
    """Draw a recombining binomial tree; ``values[n, j]`` = value after n periods with j up-moves.

    ``highlight``: optional boolean array like ``values``; True nodes are drawn in red
    (used to mark early-exercise nodes).
    """
    T = values.shape[0] - 1
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.6 * T + 3, 1.2 * T + 3))
    for n in range(T + 1):
        for j in range(n + 1):
            x, y = n, 2 * j - n
            if n < T:
                for dj in (0, 1):
                    ax.plot([x, x + 1], [y, 2 * (j + dj) - (n + 1)], "-", color="lightskyblue", lw=1, zorder=1)
            hl = highlight is not None and bool(highlight[n, j])
            ax.plot(x, y, "o", color=ASK if hl else "lightskyblue", zorder=2)
            if annotate and not np.isnan(values[n, j]):
                ax.annotate(fmt.format(values[n, j]), (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(range(T + 1)); ax.set_xlabel("Period")
    ax.set_yticks([]); ax.set_title(title)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    return _finish(fig, path) if fig else None


def plot_put_spreads_vs_strike(strikes: ArrayLike, spreads: Mapping[str, ArrayLike],
                               title: str = "Calibrated put bid-ask spread", path=None) -> Figure:
    """Spread curves over strikes (thesis Fig. 7); ``spreads`` maps a label to an array."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, (lab, sp) in zip(LINES, spreads.items()):
        ax.plot(strikes, sp, color=c, label=lab)
    ax.set(xlabel="Strike $K$", ylabel="Ask - bid", title=title); ax.legend()
    return _finish(fig, path)


def plot_replication(stock: NDArray, value: NDArray, lam0: NDArray, lam1: NDArray, residual: NDArray,
                     title: str, path=None) -> Figure:
    """Three trees: value, stock holding ``lambda1`` and self-financing residual (thesis Prop. 4.4)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plot_tree(value, f"{title}: value", ax=axes[0], fmt="{:.1f}")
    plot_tree(lam1, f"{title}: stock holding", ax=axes[1], fmt="{:.2f}")
    plot_tree(residual, f"{title}: self-financing residual", ax=axes[2], fmt="{:.2f}")
    return _finish(fig, path)


def plot_multistart_errors(tables: Mapping[str, "pd.DataFrame"], thesis_value: Optional[float] = None, path=None) -> Figure:
    """Distribution of polished calibration errors over PSO seeds (log-free strip plot)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (lab, t) in enumerate(tables.items()):
        e = t["error"].clip(upper=60)
        ax.scatter(np.full(len(e), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(e)), e, s=14, alpha=0.7)
    if thesis_value is not None:
        ax.axhline(thesis_value, color="k", ls="--", lw=1, label=f"thesis value {thesis_value:g}"); ax.legend()
    ax.set_xticks(range(len(tables))); ax.set_xticklabels(list(tables), rotation=15, ha="right")
    ax.set(ylabel="Squared error E (clipped at 60)", title="Calibration error over PSO seeds")
    return _finish(fig, path)


def plot_objective_tradeoff(df: "pd.DataFrame", path=None) -> Figure:
    """Mean model spread vs price RMSE for the objectives of ``results/objective_comparison.csv``."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for (b, g), c in zip(df.groupby("bounds"), LINES):
        ax.plot(g["rmse_price"], g["mean_model_spread"], "o-", color=c, label=f"b_d bounds {b}")
    ax.axhline(float(df["mean_market_spread"].iloc[0]), color="k", ls="--", lw=1, label="market mean spread")
    ax.set(xlabel="Price RMSE (USD per quote)", ylabel="Mean model spread (USD)", title="Price fit vs spread fit"); ax.legend(fontsize=8)
    return _finish(fig, path)
