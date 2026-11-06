"""Generalised ambiguity: symmetric and state-dependent lower masses (EXTENSION, not in the thesis).

Thesis model (``kind="thesis"``)
    ``b_u = q_u`` (the CRR up-probability, eq. 13) and ``b_d = q_d - a``; all ambiguity ``a = 1 - b_u - b_d``
    is taken from the down mass. Consequence: the call *bid* and the put *ask* both equal the CRR price, so
    ``C_bid - P_ask = S0 - K / R^T`` holds exactly and the model's spread sits one-sidedly above CRR.

Symmetric model (``kind="symmetric"``)
    ``b_u = q_u - a/2`` and ``b_d = q_d - a/2``: both lower masses are shrunk equally, so the belief set is
    centred on the CRR measure. Same number of parameters as the thesis model.

State-dependent ambiguity
    ``a(n, S) = a * exp(-0.5 * (ln(S / S0) / width)^2)``: ambiguity is largest near the money and fades in the
    wings (``width = inf`` gives a constant ``a``). Motivation: market spreads peak at the money.
    This is a hypothesis to be tested, see ``scripts/compare_ambiguity.py`` and ``results/ambiguity_comparison.csv``.

Pricing uses the general backward recursion of :mod:`dsbidask.choquet` with per-node weights, vectorised over
strikes, so it also covers non-constant ``a`` where the closed forms of :mod:`dsbidask.pricing` do not apply.
With a constant ``a`` and ``kind="thesis"`` it equals those closed forms (``tests/test_ambiguity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import differential_evolution

from .config import R_DEFAULT, T_DEFAULT
from .data import Quotes
from .pricing import b_up

Kind = Literal["thesis", "symmetric"]
Option = Literal["call", "put"]
Mode = Literal["price", "mid_spread", "inv_spread"]


def max_ambiguity(u: float, d: float, R: float, kind: Kind) -> float:
    """Largest ``a`` keeping both lower masses non-negative (``b_u, b_d >= 0``, ``b_u + b_d = 1 - a``)."""
    qu = b_up(u, d, R)
    qd = 1.0 - qu
    if kind == "thesis":
        return qd
    if kind == "symmetric":
        return 2.0 * min(qu, qd)
    raise ValueError(f"unknown kind {kind!r}")


@dataclass(frozen=True)
class Ambiguity:
    """Ambiguity specification. ``a`` is the peak total ambiguity ``1 - b_u - b_d``."""

    a: float
    kind: Kind = "symmetric"
    width: float = np.inf  # log-moneyness width of the Gaussian bump; inf = constant ambiguity

    def a_grid(self, stock: NDArray[np.float64], S0: float) -> NDArray[np.float64]:
        """Ambiguity at every node of the stock grid (NaN where the grid is NaN)."""
        if not np.isfinite(self.width):
            return np.where(np.isnan(stock), np.nan, self.a)
        return self.a * np.exp(-0.5 * (np.log(stock / S0) / self.width) ** 2)

    def masses(self, stock: NDArray[np.float64], S0: float, u: float, d: float, R: float):
        """Per-node lower masses ``(b_u, b_d)`` on the stock grid; raises if infeasible."""
        if self.a < 0 or self.a > max_ambiguity(u, d, R, self.kind) + 1e-12:
            raise ValueError(f"a={self.a:.6f} outside [0, {max_ambiguity(u, d, R, self.kind):.6f}] for kind={self.kind!r}")
        qu = b_up(u, d, R)
        a = self.a_grid(stock, S0)
        if self.kind == "thesis":
            return np.full_like(a, qu), (1.0 - qu) - a
        return qu - a / 2.0, (1.0 - qu) - a / 2.0


def stock_grid(S0: float, u: float, d: float, T: int) -> NDArray[np.float64]:
    """Recombining grid, entry ``[n, j]`` = ``S0 u^j d^(n-j)`` (NaN for ``j > n``)."""
    g = np.full((T + 1, T + 1), np.nan)
    for n in range(T + 1):
        j = np.arange(n + 1)
        g[n, : n + 1] = S0 * u**j * d ** (n - j)
    return g


def chain_bid_ask(S0: float, K: ArrayLike, option: Option, u: float, d: float, R: float, T: int,
                  amb: Ambiguity, american: bool = False) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bid/ask of calls or puts at several strikes by backward recursion with per-node masses.

    ``K = 0`` with ``option="call"`` prices the underlying itself (the thesis stores the stock as a strike-0 call).
    Returns ``(bid, ask)`` arrays of the length of ``K``.
    """
    K = np.atleast_1d(np.asarray(K, float))
    stock = stock_grid(S0, u, d, T)
    bu, bd = amb.masses(stock, S0, u, d, R)
    sign = 1.0 if option == "call" else -1.0
    payoff = lambda s: np.maximum(sign * (s[None, :] - K[:, None]), 0.0)
    bid = ask = payoff(stock[T, :])
    for n in range(T - 1, -1, -1):
        m = n + 1
        w_u, w_d = bu[n, :m][None, :], bd[n, :m][None, :]
        slack = 1.0 - w_u - w_d
        b_up_, b_dn_ = bid[:, 1 : m + 1], bid[:, :m]
        a_up_, a_dn_ = ask[:, 1 : m + 1], ask[:, :m]
        nb = (w_u * b_up_ + w_d * b_dn_ + slack * np.minimum(b_up_, b_dn_)) / R
        na = (w_u * a_up_ + w_d * a_dn_ + slack * np.maximum(a_up_, a_dn_)) / R
        if american:
            ex = payoff(stock[n, :m])
            nb, na = np.maximum(nb, ex), np.maximum(na, ex)
        bid, ask = nb, na
    return bid[:, 0], ask[:, 0]


def model_quotes(u: float, d: float, amb: Ambiguity, q: Quotes, R: float = R_DEFAULT, T: int = T_DEFAULT):
    """Model ``(call_bid, call_ask, put_bid, put_ask)`` at the quoted strikes."""
    cb, ca = chain_bid_ask(q.S0_bid, q.K_call, "call", u, d, R, T, amb)
    pb, pa = chain_bid_ask(q.S0_bid, q.K_put, "put", u, d, R, T, amb)
    return cb, ca, pb, pa


def quote_loss(xb, xa, mb, ma, mode: Mode = "price", spread_weight: float = 0.25, spread_floor: float = 0.01) -> float:
    """Same three losses as :func:`dsbidask.calibration.objective`, on arrays of model and market quotes."""
    if mode == "price":
        return float(np.sum((xb - mb) ** 2) + np.sum((xa - ma) ** 2))
    if mode == "mid_spread":
        return float(np.sum(((xb + xa) / 2 - (mb + ma) / 2) ** 2) + spread_weight * np.sum(((xa - xb) - (ma - mb)) ** 2))
    if mode == "inv_spread":
        w = 1.0 / np.maximum(ma - mb, spread_floor)
        return float(np.sum(((xb - mb) * w) ** 2) + np.sum(((xa - ma) * w) ** 2))
    raise ValueError(f"unknown mode {mode!r}")


def loss(u: float, d: float, amb: Ambiguity, q: Quotes, R: float = R_DEFAULT, T: int = T_DEFAULT, **kw) -> float:
    cb, ca, pb, pa = model_quotes(u, d, amb, q, R, T)
    return quote_loss(np.r_[cb, pb], np.r_[ca, pa], np.r_[q.C_bid, q.P_bid], np.r_[q.C_ask, q.P_ask], **kw)


def interval_report(u: float, d: float, amb: Ambiguity, q: Quotes, R: float = R_DEFAULT, T: int = T_DEFAULT
                    ) -> Dict[str, float]:
    """RMSE and interval diagnostics (strike-0 stock quote excluded).

    ``inside``: model [bid, ask] inside the market [bid, ask]; ``overlap``: the two intervals intersect;
    ``mid_in``: model mid lies in the market interval.
    """
    cb, ca, pb, pa = model_quotes(u, d, amb, q, R, T)
    keep = np.r_[q.K_call > 0, np.ones(len(q.K_put), bool)]
    xb, xa = np.r_[cb, pb][keep], np.r_[ca, pa][keep]
    mb, ma = np.r_[q.C_bid, q.P_bid][keep], np.r_[q.C_ask, q.P_ask][keep]
    mid = (xb + xa) / 2
    return dict(
        rmse_price=float(np.sqrt(np.mean(np.r_[(xb - mb) ** 2, (xa - ma) ** 2]))),
        mean_model_spread=float(np.mean(xa - xb)), mean_market_spread=float(np.mean(ma - mb)),
        inside=float(np.mean((xb >= mb - 1e-9) & (xa <= ma + 1e-9))),
        overlap=float(np.mean((xb <= ma) & (xa >= mb))),
        mid_in=float(np.mean((mid >= mb) & (mid <= ma))),
    )


def calibrate_ambiguity(q: Quotes, kind: Kind = "symmetric", state_dependent: bool = False, R: float = R_DEFAULT,
                        T: int = T_DEFAULT, seeds: Sequence[int] = (0, 1, 2), u_max: float = 1.056,
                        d_min: float = 0.90, eps: float = 1e-5, width_bounds: Tuple[float, float] = (0.01, 0.5),
                        maxiter: int = 200, popsize: int = 15, **loss_kw) -> Dict[str, float]:
    """Fit ``(u, d, f[, width])`` with differential evolution (best of several seeds).

    ``f in [0, 1]`` scales the ambiguity: ``a = f * max_ambiguity(u, d, kind)``, so every point of the box is
    feasible (``b_u, b_d >= 0`` and ``b_u + b_d <= 1``) and no penalty or constraint is needed.
    Free ``d`` and wide ambiguity bounds: this is *not* the thesis box (``b_d in [0.3, 0.6]``).
    """
    def unpack(x):
        u, d, f = x[0], x[1], x[2]
        w = x[3] if state_dependent else np.inf
        return u, d, Ambiguity(f * max_ambiguity(u, d, R, kind), kind, w)

    def f_obj(x):
        u, d, amb = unpack(x)
        return loss(u, d, amb, q, R, T, **loss_kw)

    bounds = [(R + eps, u_max), (d_min, R - eps), (0.0, 1.0)] + ([width_bounds] if state_dependent else [])
    best = None
    for s in seeds:
        r = differential_evolution(f_obj, bounds, seed=s, maxiter=maxiter, popsize=popsize, tol=1e-10, polish=True)
        if best is None or r.fun < best.fun:
            best = r
    u, d, amb = unpack(best.x)
    qu = b_up(u, d, R)
    bu, bd = (qu, 1 - qu - amb.a) if kind == "thesis" else (qu - amb.a / 2, 1 - qu - amb.a / 2)
    return dict(kind=kind, state_dependent=state_dependent, u=float(u), d=float(d), a=float(amb.a),
                width=float(amb.width), b_u=float(bu), b_d=float(bd), error=float(best.fun))
