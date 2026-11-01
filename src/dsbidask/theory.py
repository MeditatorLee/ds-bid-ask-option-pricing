"""Theoretical spreads (thesis Sec. 4.1 Fig. 1 and Sec. 5 Fig. 7)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .choquet import american_put_bid_ask
from .pricing import b_up, european_call_bid_ask, european_put_bid_ask, european_puts_bid_ask


def spread_vs_epsilon(kind: str, S0: float, K: float, u: float, d: float, R: float, T: int,
                      eps: Optional[ArrayLike] = None) -> Tuple[NDArray, NDArray]:
    """Ask - bid of a European ``'call'``/``'put'`` as a function of epsilon (Prop. 4.3, Fig. 1).

    ``b_d = epsilon * (1 - b_u)``: epsilon = 1 gives ``b_u + b_d = 1`` (zero spread, CRR); the spread is a
    polynomial of degree ``T`` in epsilon. The thesis admits ``epsilon in (0, 1]``; the default grid
    here is ``[0, 1)`` as in the original code (see ``docs/THESIS_ERRATA.md``, C6).
    """
    eps = np.arange(0, 1, 0.01) if eps is None else np.asarray(eps, float)
    bu = b_up(u, d, R)
    price = {"call": european_call_bid_ask, "put": european_put_bid_ask}[kind]
    return eps, np.array([(lambda b, a: a - b)(*price(S0, K, u, d, e * (1 - bu), R, T)) for e in eps])


def put_spread_vs_strike(S0: float, strikes: ArrayLike, u: float, d: float, b_d: float, R: float, T: int,
                         american: bool = False) -> Tuple[NDArray, NDArray, NDArray]:
    """Bid, ask and spread of puts over a strike grid (Fig. 7): European closed form or American recursion."""
    strikes = np.asarray(strikes, float)
    if american:
        res = np.array([american_put_bid_ask(S0, K, u, d, b_d, R, T) for K in strikes])
        bid, ask = res[:, 0], res[:, 1]
    else:
        bid, ask = european_puts_bid_ask(S0, strikes, u, d, b_d, R, T)
    return bid, ask, ask - bid
