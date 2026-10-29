"""General bid/ask pricer: backward recursion with one-step Choquet expectations.

Works for ANY payoff (monotone or not) and for early exercise. Thesis eqs. (18)-(21):

    bid:  Y_n    =  (1/R) * C[ Y_{n+1} | S_n ]
    ask:  Ybar_n = -(1/R) * C[-Ybar_{n+1} | S_n]

The one-step belief puts mass b_u on {up}, b_d on {down} and 1 - b_u - b_d on {up, down}, so

    C[V]  = b_u V_u + b_d V_d + (1 - b_u - b_d) * min(V_u, V_d)      (lower Choquet expectation)
    -C[-V] = b_u V_u + b_d V_d + (1 - b_u - b_d) * max(V_u, V_d)     (upper Choquet expectation)

For monotone payoffs this coincides with the closed forms of :mod:`dsbidask.pricing`.
"""
from __future__ import annotations
import numpy as np
from .pricing import b_up


def choquet_lower(V_up, V_down, b_u, b_d):
    """One-step lower Choquet expectation of (V_up, V_down)."""
    return b_u * V_up + b_d * V_down + (1.0 - b_u - b_d) * np.minimum(V_up, V_down)


def choquet_upper(V_up, V_down, b_u, b_d):
    """One-step upper Choquet expectation (dual of the lower one)."""
    return b_u * V_up + b_d * V_down + (1.0 - b_u - b_d) * np.maximum(V_up, V_down)


def bid_ask_trees(payoff, S0, u, d, b_d, R, T, american: bool = False):
    """Bid/ask value trees for an arbitrary payoff by backward induction.

    Parameters
    ----------
    payoff : callable mapping an array of stock prices to payoffs.
    american : if True, take max(exercise value, continuation) at every node.
        (The thesis treats European claims; the American extension uses the same
        one-step lower/upper expectations and is an assumption, not a thesis result.)

    Returns
    -------
    stock, bid, ask : arrays of shape (T+1, T+1); entry [n, j] = node after n periods, j up-moves.
    """
    bu = b_up(u, d, R)
    if bu + b_d > 1 + 1e-12:
        raise ValueError(f"need b_u + b_d <= 1, got {bu + b_d:.6f}")
    stock = np.full((T + 1, T + 1), np.nan)
    for n in range(T + 1):
        j = np.arange(n + 1)
        stock[n, : n + 1] = S0 * u**j * d ** (n - j)
    bid, ask = np.full_like(stock, np.nan), np.full_like(stock, np.nan)
    bid[T, :] = ask[T, :] = payoff(stock[T, :])
    for n in range(T - 1, -1, -1):
        m = n + 1
        cb = choquet_lower(bid[n + 1, 1 : m + 1], bid[n + 1, :m], bu, b_d) / R
        ca = choquet_upper(ask[n + 1, 1 : m + 1], ask[n + 1, :m], bu, b_d) / R
        if american:
            ex = payoff(stock[n, :m])
            cb, ca = np.maximum(cb, ex), np.maximum(ca, ex)
        bid[n, :m], ask[n, :m] = cb, ca
    return stock, bid, ask


def bid_ask(payoff, S0, u, d, b_d, R, T, american: bool = False):
    """Root bid/ask price of an arbitrary payoff. Returns ``(bid, ask)``."""
    _, bid, ask = bid_ask_trees(payoff, S0, u, d, b_d, R, T, american)
    return float(bid[0, 0]), float(ask[0, 0])


def american_put_bid_ask(S0, K, u, d, b_d, R, T, return_trees: bool = False):
    """American put bid/ask (early exercise allowed). Returns ``(bid, ask)`` or the trees too."""
    pay = lambda s: np.maximum(K - s, 0.0)
    out = bid_ask_trees(pay, S0, u, d, b_d, R, T, american=True)
    return out if return_trees else (float(out[1][0, 0]), float(out[2][0, 0]))


def early_exercise_nodes(payoff, S0, u, d, b_d, R, T):
    """Boolean tree (bid, ask): True where exercising now beats continuing (and payoff > 0)."""
    stock, bid_a, ask_a = bid_ask_trees(payoff, S0, u, d, b_d, R, T, american=True)
    bu = b_up(u, d, R)
    flags = []
    for tree, f in ((bid_a, choquet_lower), (ask_a, choquet_upper)):
        flag = np.zeros_like(stock, dtype=bool)
        for n in range(T):
            m = n + 1
            cont = f(tree[n + 1, 1 : m + 1], tree[n + 1, :m], bu, b_d) / R
            ex = payoff(stock[n, :m])
            flag[n, :m] = (ex > cont) & (ex > 0)
        flags.append(flag)
    return tuple(flags)
