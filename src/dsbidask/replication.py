"""Replicating portfolios for the bid/ask price processes (thesis Section 4.2, eqs. 26-29, Prop. 4.4).

For every node ``(n, j)`` we solve, with bond ``B_n = R^n`` and the up/down successor nodes,

    lambda0 * B_{n+1} + lambda1 * S_{n+1} = Y_{n+1}      (eq. 26; eq. 27 for the ask)

and report the self-financing residual ``Y_n - (lambda0 * B_n + lambda1 * S_n)`` (eqs. 28-29).
Prop. 4.4: the bid strategy is self-financing (residual 0) iff ``lambda1 >= 0`` (non-decreasing
payoff); the ask strategy iff ``delta1 <= 0`` (non-increasing payoff).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

Tree = NDArray[np.float64]


def replicating_portfolio(stock: Tree, value: Tree, R: float) -> Tuple[Tree, Tree, Tree]:
    """Replicating (bond, stock) holdings and self-financing residual for a value tree.

    ``stock`` and ``value`` have shape ``(T+1, T+1)`` with ``[n, j]`` = n periods, j up-moves.
    Returns ``(lam0, lam1, residual)`` of the same shape (NaN at the terminal layer).
    """
    T = stock.shape[0] - 1
    lam0 = np.full_like(stock, np.nan); lam1 = lam0.copy(); res = lam0.copy()
    for n in range(T):
        m = n + 1
        Su, Sd = stock[n + 1, 1 : m + 1], stock[n + 1, :m]
        Yu, Yd = value[n + 1, 1 : m + 1], value[n + 1, :m]
        l1 = (Yu - Yd) / (Su - Sd)
        l0 = (Yu - l1 * Su) / R ** (n + 1)
        lam1[n, :m], lam0[n, :m] = l1, l0
        res[n, :m] = value[n, :m] - (l0 * R**n + l1 * stock[n, :m])
    return lam0, lam1, res


def is_self_financing(residual: Tree, tol: float = 1e-9) -> bool:
    """True if the self-financing residual vanishes everywhere (eqs. 28-29 hold with equality)."""
    return bool(np.nanmax(np.abs(residual)) < tol)
