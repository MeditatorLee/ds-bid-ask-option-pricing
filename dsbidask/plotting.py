"""Plotting helpers (English labels, no font dependencies)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .pricing import european_call_bid_ask, european_put_bid_ask

BID, ASK = "#32CD32", "#DC143C"
LINES = ["#4B8BBE", "#FFB31A", "#32CD32", "#DC143C"]


def _finish(fig, path):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300)
    return fig


def plot_market_quotes(K, bid, ask, title, path=None):
    """Market bid/ask prices and spreads against strike (two panels)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.scatter(K, ask, marker=".", color=ASK, label="Ask")
    a1.scatter(K, bid, marker=".", color=BID, label="Bid")
    a1.set(xlabel="Strike", ylabel="Price", title=f"{title}: bid and ask"); a1.legend()
    a2.scatter(K, np.asarray(ask) - np.asarray(bid), marker="+", color=ASK)
    a2.set(xlabel="Strike", ylabel="Ask - bid", title=f"{title}: spread")
    return _finish(fig, path)


def plot_spread_vs_epsilon(curves, ylabel, title, path=None):
    """curves: dict {label: (eps, spread)}."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, (lab, (e, s)) in zip(LINES, curves.items()):
        ax.plot(e, s, color=c, label=lab)
    ax.set(xlabel=r"$\epsilon$", ylabel=ylabel, title=title); ax.legend()
    return _finish(fig, path)


def plot_calibration_fit(q, params, R, T, path=None):
    """Model bid/ask curves vs market quotes for calls and puts."""
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


def plot_tree(values, title="", path=None, fmt="{:.2f}", ax=None, annotate=True):
    """Draw a recombining binomial tree. values[n, j] = value after n periods with j up-moves."""
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
            ax.plot(x, y, "o", color="lightskyblue", zorder=2)
            if annotate:
                ax.annotate(fmt.format(values[n, j]), (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(range(T + 1)); ax.set_xlabel("Period")
    ax.set_yticks([]); ax.set_title(title)
    for s in ("left", "right", "top"): ax.spines[s].set_visible(False)
    return _finish(fig, path) if fig else None
