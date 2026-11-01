"""Closed-form pricing under the DS multiplicative random walk (thesis Prop. 4.1-4.2).

Notation (thesis Chapter 3-4): up factor ``u``, down factor ``d``, gross one-period rate
``R = 1 + r`` and ``b_d`` (lower mass of the down move). ``b_u = (R - d)/(u - d)`` follows from the
one-step Choquet martingale condition (eq. 13); the model needs ``b_u + b_d <= 1`` and equality
recovers CRR.

For *monotone* payoffs the Choquet bid/ask reduce to binomial sums (Prop. 4.1, 4.2):

* non-decreasing payoff (call): bid weights ``(b_u, 1 - b_u)`` (eq. 22), ask ``(1 - b_d, b_d)`` (eq. 23)
* non-increasing payoff (put):  bid weights ``(1 - b_d, b_d)`` (eq. 24), ask ``(b_u, 1 - b_u)`` (eq. 25)

Non-monotone payoffs (e.g. a straddle) are NOT covered by these formulas; use
:mod:`dsbidask.choquet` (backward recursion, eqs. 19 and 21) for them.
"""
from __future__ import annotations

from typing import Callable, Literal, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import comb

Payoff = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""A contract function ``phi`` (eq. 17) mapping terminal stock prices to payoffs."""
Monotone = Literal["increasing", "decreasing"]


def b_up(u: float, d: float, R: float) -> float:
    """Lower mass of the up move, ``b_u = (R - d)/(u - d)`` (thesis eq. 13)."""
    return (R - d) / (u - d)


def terminal_weights(p: float, T: int) -> NDArray[np.float64]:
    """Binomial weights ``C(T,k) p^k (1-p)^(T-k)``, ``k = 0..T`` (the sums in eqs. 22-25)."""
    k = np.arange(T + 1)
    return comb(T, k) * p**k * (1.0 - p) ** (T - k)


def terminal_prices(S0: float, u: float, d: float, T: int) -> NDArray[np.float64]:
    """Terminal stock prices ``u^k d^(T-k) S0``, ``k`` = number of up-moves."""
    k = np.arange(T + 1)
    return S0 * u**k * d ** (T - k)


def _weights_for(monotone: Monotone, u: float, d: float, b_d: float, R: float, T: int):
    bu = b_up(u, d, R)
    if monotone == "increasing":
        pb, pa = bu, 1.0 - b_d
    elif monotone == "decreasing":
        pb, pa = 1.0 - b_d, bu
    else:
        raise ValueError("monotone must be 'increasing' or 'decreasing'")
    return terminal_weights(pb, T), terminal_weights(pa, T)


def european_bid_ask(payoff: Payoff, S0: float, u: float, d: float, b_d: float, R: float, T: int,
                     monotone: Monotone) -> Tuple[float, float]:
    """Bid/ask of a European claim with a *monotone* payoff (Prop. 4.1 if increasing, 4.2 if decreasing).

    Returns ``(bid, ask)``. Misusing this for a non-monotone payoff silently returns invalid prices
    (e.g. bid > ask); see ``tests/test_choquet.py``.
    """
    wb, wa = _weights_for(monotone, u, d, b_d, R, T)
    pay = payoff(terminal_prices(S0, u, d, T))
    return float(wb @ pay / R**T), float(wa @ pay / R**T)


def european_call_bid_ask(S0: float, K: float, u: float, d: float, b_d: float, R: float, T: int) -> Tuple[float, float]:
    """European call ``(bid, ask)`` (eqs. 22-23)."""
    return european_bid_ask(lambda s: np.maximum(s - K, 0.0), S0, u, d, b_d, R, T, "increasing")


def european_put_bid_ask(S0: float, K: float, u: float, d: float, b_d: float, R: float, T: int) -> Tuple[float, float]:
    """European put ``(bid, ask)`` (eqs. 24-25)."""
    return european_bid_ask(lambda s: np.maximum(K - s, 0.0), S0, u, d, b_d, R, T, "decreasing")


def european_calls_bid_ask(S0: float, K: ArrayLike, u: float, d: float, b_d: float, R: float, T: int
                           ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorised eqs. 22-23 over an array of strikes. Returns ``(bid, ask)`` arrays."""
    K = np.asarray(K, float)
    wb, wa = _weights_for("increasing", u, d, b_d, R, T)
    pay = np.maximum(terminal_prices(S0, u, d, T)[None, :] - K[:, None], 0.0)
    return pay @ wb / R**T, pay @ wa / R**T


def european_puts_bid_ask(S0: float, K: ArrayLike, u: float, d: float, b_d: float, R: float, T: int
                          ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorised eqs. 24-25 over an array of strikes. Returns ``(bid, ask)`` arrays."""
    K = np.asarray(K, float)
    wb, wa = _weights_for("decreasing", u, d, b_d, R, T)
    pay = np.maximum(K[:, None] - terminal_prices(S0, u, d, T)[None, :], 0.0)
    return pay @ wb / R**T, pay @ wa / R**T


def crr_price(payoff: Payoff, S0: float, u: float, d: float, R: float, T: int) -> float:
    """Classical CRR price: the ``b_u + b_d = 1`` case (zero ambiguity, Prop. 4.3 at epsilon = 1)."""
    w = terminal_weights(b_up(u, d, R), T)
    return float(w @ payoff(terminal_prices(S0, u, d, T)) / R**T)


def european_tree(payoff: Payoff, S0: float, u: float, d: float, b_d: float, R: float, T: int,
                  monotone: Monotone):
    """Node-wise bid/ask value trees of a monotone European claim (Prop. 4.1-4.2 at every node ``s_n``).

    Returns ``(stock, bid, ask)``, each of shape ``(T+1, T+1)``; entry ``[n, j]`` is the node after
    ``n`` periods with ``j`` up-moves (NaN where ``j > n``).
    """
    stock = np.full((T + 1, T + 1), np.nan)
    bid, ask = stock.copy(), stock.copy()
    for n in range(T + 1):
        for j in range(n + 1):
            s = S0 * u**j * d ** (n - j)
            stock[n, j] = s
            bid[n, j], ask[n, j] = european_bid_ask(payoff, s, u, d, b_d, R, T - n, monotone)
    return stock, bid, ask
