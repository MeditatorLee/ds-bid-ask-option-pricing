"""Theoretical spreads (thesis Sec. 4.1 Fig. 1 and Sec. 5 Fig. 7)."""
from __future__ import annotations
import numpy as np
from .pricing import b_up, european_call_bid_ask, european_put_bid_ask, european_puts_bid_ask
from .choquet import american_put_bid_ask


def spread_vs_epsilon(kind, S0, K, u, d, R, T, eps=None):
    """Ask - bid of a European call/put as a function of epsilon, where b_d = epsilon * (1 - b_u).

    epsilon = 1 gives b_u + b_d = 1 (zero spread, CRR); epsilon -> 0 gives the widest spread.
    (Prop. 4.3: the spread is a polynomial of degree T in epsilon.)
    """
    eps = np.arange(0, 1, 0.01) if eps is None else np.asarray(eps, float)
    bu = b_up(u, d, R)
    price = {"call": european_call_bid_ask, "put": european_put_bid_ask}[kind]
    return eps, np.array([(lambda b, a: a - b)(*price(S0, K, u, d, e * (1 - bu), R, T)) for e in eps])


def put_spread_vs_strike(S0, strikes, u, d, b_d, R, T, american=False):
    """Bid, ask and spread of puts over a strike grid (European closed form or American recursion)."""
    strikes = np.asarray(strikes, float)
    if american:
        res = np.array([american_put_bid_ask(S0, K, u, d, b_d, R, T) for K in strikes])
        bid, ask = res[:, 0], res[:, 1]
    else:
        bid, ask = european_puts_bid_ask(S0, strikes, u, d, b_d, R, T)
    return bid, ask, ask - bid
