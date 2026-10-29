"""Closed-form pricing under the DS multiplicative random walk (thesis Prop. 4.1-4.2).

Parameters: up factor ``u``, down factor ``d``, gross one-period rate ``R = 1 + r`` and ``b_d``
(lower mass of the down move). ``b_u = (R - d)/(u - d)``; the model needs ``b_u + b_d <= 1`` and
equality recovers CRR.

For *monotone* payoffs the Choquet bid/ask reduce to binomial sums:

* non-decreasing payoff (call): bid weights (b_u, 1 - b_u), ask weights (1 - b_d, b_d)
* non-increasing payoff (put):  bid weights (1 - b_d, b_d), ask weights (b_u, 1 - b_u)

Non-monotone payoffs (e.g. a straddle) are NOT covered by these formulas. Use
:mod:`dsbidask.choquet` (backward recursion, thesis eqs. 19 and 21) for them.
"""
from __future__ import annotations
import numpy as np
from scipy.special import comb


def b_up(u: float, d: float, R: float) -> float:
    """Lower mass of the up move implied by the one-step Choquet martingale condition."""
    return (R - d) / (u - d)


def terminal_weights(p: float, T: int) -> np.ndarray:
    """Binomial weights C(T,k) p^k (1-p)^(T-k), k = 0..T."""
    k = np.arange(T + 1)
    return comb(T, k) * p**k * (1.0 - p) ** (T - k)


def terminal_prices(S0, u: float, d: float, T: int) -> np.ndarray:
    """Terminal stock prices u^k d^(T-k) S0, k = 0..T (k = number of up-moves)."""
    k = np.arange(T + 1)
    return S0 * u**k * d ** (T - k)


def _weights_for(monotone: str, u, d, b_d, R, T):
    bu = b_up(u, d, R)
    if monotone == "increasing":
        pb, pa = bu, 1.0 - b_d
    elif monotone == "decreasing":
        pb, pa = 1.0 - b_d, bu
    else:
        raise ValueError("monotone must be 'increasing' or 'decreasing'")
    return terminal_weights(pb, T), terminal_weights(pa, T)


def european_bid_ask(payoff, S0, u, d, b_d, R, T, monotone: str):
    """Bid/ask of a European claim with a monotone payoff.

    ``monotone`` is 'increasing' or 'decreasing'. Returns ``(bid, ask)``.
    """
    wb, wa = _weights_for(monotone, u, d, b_d, R, T)
    pay = payoff(terminal_prices(S0, u, d, T))
    return float(wb @ pay / R**T), float(wa @ pay / R**T)


def european_call_bid_ask(S0, K, u, d, b_d, R, T):
    return european_bid_ask(lambda s: np.maximum(s - K, 0.0), S0, u, d, b_d, R, T, "increasing")


def european_put_bid_ask(S0, K, u, d, b_d, R, T):
    return european_bid_ask(lambda s: np.maximum(K - s, 0.0), S0, u, d, b_d, R, T, "decreasing")


def european_calls_bid_ask(S0, K, u, d, b_d, R, T):
    """Vectorised over an array of strikes ``K``. Returns ``(bid, ask)`` arrays."""
    K = np.asarray(K, float)
    wb, wa = _weights_for("increasing", u, d, b_d, R, T)
    pay = np.maximum(terminal_prices(S0, u, d, T)[None, :] - K[:, None], 0.0)
    return pay @ wb / R**T, pay @ wa / R**T


def european_puts_bid_ask(S0, K, u, d, b_d, R, T):
    """Vectorised over an array of strikes ``K``. Returns ``(bid, ask)`` arrays."""
    K = np.asarray(K, float)
    wb, wa = _weights_for("decreasing", u, d, b_d, R, T)
    pay = np.maximum(K[:, None] - terminal_prices(S0, u, d, T)[None, :], 0.0)
    return pay @ wb / R**T, pay @ wa / R**T


def crr_price(payoff, S0, u, d, R, T):
    """Classical CRR price (the b_u + b_d = 1 case)."""
    w = terminal_weights(b_up(u, d, R), T)
    return float(w @ payoff(terminal_prices(S0, u, d, T)) / R**T)


def european_tree(payoff, S0, u, d, b_d, R, T, monotone: str):
    """Node-wise bid/ask value trees of a monotone European claim (thesis Prop. 4.1-4.2).

    Returns ``(stock, bid, ask)``, each of shape (T+1, T+1); entry ``[n, j]`` is the node after
    ``n`` periods with ``j`` up-moves (NaN where j > n).
    """
    stock = np.full((T + 1, T + 1), np.nan)
    bid, ask = stock.copy(), stock.copy()
    for n in range(T + 1):
        for j in range(n + 1):
            s = S0 * u**j * d ** (n - j)
            stock[n, j] = s
            bid[n, j], ask[n, j] = european_bid_ask(payoff, s, u, d, b_d, R, T - n, monotone)
    return stock, bid, ask
