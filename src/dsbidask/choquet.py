"""General bid/ask pricer: backward recursion with one-step Choquet expectations.

Works for ANY payoff (monotone or not) and, as an extension, for early exercise.
Thesis eqs. (18)-(21), Markov form:

    bid:  Y_n    =  (1/R) * C[ Y_{n+1} | S_n ]                       (eq. 19)
    ask:  Ybar_n = -(1/R) * C[-Ybar_{n+1} | S_n]                     (eq. 21)

The one-step belief (eq. 13) puts mass ``b_u`` on {up}, ``b_d`` on {down} and ``1 - b_u - b_d`` on
{up, down}, so for values ``V_u``, ``V_d``:

    C[V]   = b_u V_u + b_d V_d + (1 - b_u - b_d) * min(V_u, V_d)    (lower Choquet expectation)
    -C[-V] = b_u V_u + b_d V_d + (1 - b_u - b_d) * max(V_u, V_d)    (upper Choquet expectation)

For monotone payoffs this coincides with the closed forms of :mod:`dsbidask.pricing`
(checked in ``tests/test_choquet.py``).

American extension (NOT in the thesis; see ``docs/AMERICAN.md``)
----------------------------------------------------------------
With ``american=True`` each node takes ``max(exercise value, continuation value)``, using the lower
expectation for the bid tree and the upper expectation for the ask tree. For a put the continuation
values stay non-increasing in the stock price, so this equals the fixed-weight recursion of the
original authors' ``AmericanPut.py`` (bid weights ``(1-b_d, b_d)``, ask weights ``(b_u, 1-b_u)``).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .pricing import Payoff, b_up

Tree = NDArray[np.float64]


def choquet_lower(V_up: Tree, V_down: Tree, b_u: float, b_d: float) -> Tree:
    """One-step lower Choquet expectation ``C[V]`` of ``(V_up, V_down)`` (eq. 18-19)."""
    return b_u * V_up + b_d * V_down + (1.0 - b_u - b_d) * np.minimum(V_up, V_down)


def choquet_upper(V_up: Tree, V_down: Tree, b_u: float, b_d: float) -> Tree:
    """One-step upper Choquet expectation ``-C[-V]``, the dual of the lower one (eq. 21)."""
    return b_u * V_up + b_d * V_down + (1.0 - b_u - b_d) * np.maximum(V_up, V_down)


def bid_ask_trees(payoff: Payoff, S0: float, u: float, d: float, b_d: float, R: float, T: int,
                  american: bool = False) -> Tuple[Tree, Tree, Tree]:
    """Bid/ask value trees for an arbitrary payoff by backward induction (eqs. 19 and 21).

    Parameters
    ----------
    payoff : contract function ``phi`` (eq. 17) acting on an array of stock prices.
    american : if True, take ``max(exercise value, continuation)`` at every node (extension, see module doc).

    Returns
    -------
    stock, bid, ask : arrays of shape ``(T+1, T+1)``; entry ``[n, j]`` = node after ``n`` periods, ``j`` up-moves.
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


def bid_ask(payoff: Payoff, S0: float, u: float, d: float, b_d: float, R: float, T: int,
            american: bool = False) -> Tuple[float, float]:
    """Root ``(bid, ask)`` of an arbitrary payoff (see :func:`bid_ask_trees`)."""
    _, bid, ask = bid_ask_trees(payoff, S0, u, d, b_d, R, T, american)
    return float(bid[0, 0]), float(ask[0, 0])


def american_put_bid_ask(S0: float, K: float, u: float, d: float, b_d: float, R: float, T: int,
                         return_trees: bool = False):
    """American put ``(bid, ask)``; with ``return_trees`` returns ``(stock, bid_tree, ask_tree)``."""
    pay = lambda s: np.maximum(K - s, 0.0)
    out = bid_ask_trees(pay, S0, u, d, b_d, R, T, american=True)
    return out if return_trees else (float(out[1][0, 0]), float(out[2][0, 0]))


def early_exercise_nodes(payoff: Payoff, S0: float, u: float, d: float, b_d: float, R: float, T: int
                         ) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Boolean trees ``(bid, ask)``: True where exercising now beats continuing (and payoff > 0)."""
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
    return flags[0], flags[1]
