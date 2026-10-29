"""Theoretical bid-ask spreads as functions of the model parameters (thesis, Sec. 4.1)."""
from __future__ import annotations
import numpy as np
from .pricing import b_up, european_call_bid_ask, european_put_bid_ask


def spread_vs_epsilon(kind, S0, K, u, d, R, T, eps=None):
    """Ask - bid of a European call/put as a function of epsilon, where b_d = epsilon * (1 - b_u).

    epsilon = 1 gives b_u + b_d = 1 (zero spread, CRR); epsilon = 0 gives the widest spread.
    """
    eps = np.arange(0, 1, 0.01) if eps is None else np.asarray(eps)
    bu = b_up(u, d, R)
    price = {"call": european_call_bid_ask, "put": european_put_bid_ask}[kind]
    out = []
    for e in eps:
        b, a = price(S0, K, u, d, e * (1 - bu), R, T)
        out.append(a - b)
    return eps, np.array(out)
