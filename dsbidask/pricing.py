"""Pricing under the DS multiplicative random walk.

Parameters: up factor u, down factor d, gross rate R, and b_d (lower mass of the down move).
b_u = (R - d)/(u - d). The model needs b_u + b_d <= 1; equality recovers CRR.
For *monotone* payoffs the Choquet bid/ask reduce to binomial sums:
  non-decreasing payoff (call): bid weights (b_u, 1-b_u), ask weights (1-b_d, b_d)
  non-increasing payoff (put):  bid weights (1-b_d, b_d), ask weights (b_u, 1-b_u)
Non-monotone payoffs (e.g. straddle) are NOT covered by these formulas.
"""
from __future__ import annotations
import numpy as np
from scipy.special import comb


def b_up(u: float, d: float, R: float) -> float:
    return (R - d) / (u - d)


def _terminal_sum(p, payoff, S0, u, d, R, T):
    k = np.arange(T + 1)
    w = comb(T, k) * p**k * (1 - p) ** (T - k)
    return float(np.sum(w * payoff(u**k * d ** (T - k) * S0)) / R**T)


def european_bid_ask(payoff, S0, u, d, b_d, R, T, monotone: str):
    """Bid/ask of a European claim. monotone: 'increasing' or 'decreasing'."""
    bu = b_up(u, d, R)
    if monotone == "increasing":
        pb, pa = bu, 1 - b_d
    elif monotone == "decreasing":
        pb, pa = 1 - b_d, bu
    else:
        raise ValueError("monotone must be 'increasing' or 'decreasing'")
    return _terminal_sum(pb, payoff, S0, u, d, R, T), _terminal_sum(pa, payoff, S0, u, d, R, T)


def european_call_bid_ask(S0, K, u, d, b_d, R, T):
    return european_bid_ask(lambda s: np.maximum(s - K, 0.0), S0, u, d, b_d, R, T, "increasing")


def european_put_bid_ask(S0, K, u, d, b_d, R, T):
    return european_bid_ask(lambda s: np.maximum(K - s, 0.0), S0, u, d, b_d, R, T, "decreasing")


def crr_price(payoff, S0, u, d, R, T):
    """Classical CRR price (b_u + b_d = 1 case)."""
    return _terminal_sum(b_up(u, d, R), payoff, S0, u, d, R, T)


def american_put_bid_ask(S0, K, u, d, b_d, R, T, return_trees: bool = False):
    """Backward induction with early exercise; returns (bid, ask).

    Bid tree uses weights (1-b_d, b_d) and ask tree (b_u, 1-b_u), mirroring the European put.
    Node (n, j) = stock after n steps with j up-moves.
    """
    bu = b_up(u, d, R)
    out, trees, flags = [], [], []
    for p in (1 - b_d, bu):
        S = lambda n: S0 * u ** np.arange(n + 1) * d ** (n - np.arange(n + 1))
        V = np.maximum(K - S(T), 0.0)
        tree, flag = [V.copy()], [np.zeros(T + 1, bool)]
        for n in range(T - 1, -1, -1):
            cont = (p * V[1:] + (1 - p) * V[:-1]) / R
            ex = np.maximum(K - S(n), 0.0)
            V = np.maximum(cont, ex)
            tree.append(V.copy()); flag.append(ex > cont)
        out.append(float(V[0])); trees.append(tree[::-1]); flags.append(flag[::-1])
    return (out[0], out[1], trees, flags) if return_trees else (out[0], out[1])


def european_tree(payoff, S0, u, d, b_d, R, T, monotone: str):
    """Node-wise bid/ask value trees of a European claim with monotone payoff.

    Returns (stock, bid, ask), each of shape (T+1, T+1); entry [n, j] is the node after
    n periods with j up-moves (entries with j > n are NaN). At every node the value is the
    bid/ask of the claim with the remaining T-n periods, started from that node's stock price.
    """
    stock = np.full((T + 1, T + 1), np.nan)
    bid, ask = stock.copy(), stock.copy()
    for n in range(T + 1):
        for j in range(n + 1):
            s = S0 * u**j * d ** (n - j)
            stock[n, j] = s
            bid[n, j], ask[n, j] = european_bid_ask(payoff, s, u, d, b_d, R, T - n, monotone)
    return stock, bid, ask
