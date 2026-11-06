"""Continuous-time limit: the DS geometric Brownian motion (diffusive scaling). EXTENSION, not in the thesis.

Definition
----------
Take the DS-multiplicative random walk with ``n`` steps of length ``D = T/n``, ``u = exp(sigma sqrt(D))``, ``d = 1/u``,
``R = exp(r D)`` and ambiguity ``eps_n = 1 - b_u - b_d = theta sqrt(D) / sigma`` (``theta >= 0``, per year). The
**DS-GBM** is the limit ``n -> infinity`` of the recursions (19) and (21) of the thesis. Two parametrisations of the
one-step masses are supported (``q`` is the CRR up-probability):

* ``"thesis"``    : ``b_u = q``,           ``b_d = 1 - q - eps``  (central drift offset ``m_c = theta``)
* ``"symmetric"`` : ``b_u = q - eps/2``,   ``b_d = 1 - q - eps/2`` (``m_c = 0``)

Results (proofs in ``docs/CONTINUOUS_LIMIT.md``)
------------------------------------------------
* Only ``eps_n ~ theta sqrt(D)`` gives a non-degenerate limit (a larger ambiguity makes the ask diverge, a smaller one
  gives Black-Scholes).
* Monotone payoffs: the bid/ask are Black-Scholes prices at a *shifted spot* ``S exp(m T)`` (:func:`ds_gbm_bid_ask`).
* Any payoff: the limit solves ``v_tau = s^2/2 v_xx + (r + m_c - s^2/2) v_x +/- theta |v_x| - r v`` (``x = ln S``,
  ``+`` ask, ``-`` bid), solved numerically by :func:`pde_bid_ask`.

Consequence tested in ``scripts/continuous_market_test.py``: every option spread is bounded by the underlying's spread.
"""
from __future__ import annotations

from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import differential_evolution
from scipy.stats import norm

from .config import R_ANNUAL, T_DEFAULT
from .data import Quotes

Kind = Literal["thesis", "symmetric"]
Option = Literal["call", "put"]
Payoff = Callable[[NDArray[np.float64]], NDArray[np.float64]]

R_CONT: float = float(np.log(1.0 + R_ANNUAL))
"""Continuously compounded annual rate matching ``R_DEFAULT`` (``R_DEFAULT ** 250 = exp(R_CONT)``)."""
T_YEARS: float = T_DEFAULT / 250.0
"""Default maturity in years (20 trading days, 250 per year)."""

# Spot shift ``m`` in units of ``theta`` as (bid, ask): the price is Black-Scholes at spot ``S exp(m * theta * T)``.
_SHIFT = {("thesis", "call"): (0.0, 2.0), ("thesis", "put"): (2.0, 0.0),
          ("symmetric", "call"): (-1.0, 1.0), ("symmetric", "put"): (1.0, -1.0)}


def _out(x: NDArray, scalar: bool):
    return float(x[0]) if scalar else x


def bs_call(S: ArrayLike, K: ArrayLike, r: float, sigma: float, T: float):
    """Black-Scholes call (no dividends). ``K = 0`` returns ``S`` (the underlying as a strike-0 call)."""
    scalar = np.ndim(S) == 0 and np.ndim(K) == 0
    S, K = np.broadcast_arrays(np.atleast_1d(np.asarray(S, float)), np.atleast_1d(np.asarray(K, float)))
    out = S.astype(float).copy()
    k = K > 0
    d1 = (np.log(S[k] / K[k]) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    out[k] = S[k] * norm.cdf(d1) - K[k] * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))
    return _out(out, scalar)


def bs_put(S: ArrayLike, K: ArrayLike, r: float, sigma: float, T: float):
    """Black-Scholes put via parity."""
    scalar = np.ndim(S) == 0 and np.ndim(K) == 0
    c = np.atleast_1d(bs_call(S, K, r, sigma, T))
    S_, K_ = np.broadcast_arrays(np.atleast_1d(np.asarray(S, float)), np.atleast_1d(np.asarray(K, float)))
    return _out(c - S_ + K_ * np.exp(-r * T), scalar)


def shift(kind: Kind, option: Option) -> Tuple[float, float]:
    """``(m_bid, m_ask)`` in units of ``theta`` (see module doc)."""
    return _SHIFT[(kind, option)]


def ds_gbm_bid_ask(S0: ArrayLike, K: ArrayLike, option: Option, sigma: float, theta: float, T: float = T_YEARS,
                   r: float = R_CONT, kind: Kind = "symmetric"):
    """Closed-form DS-GBM ``(bid, ask)`` of a European call or put (Theorem 1 of ``docs/CONTINUOUS_LIMIT.md``).

    ``theta = 0`` gives ``bid = ask = `` Black-Scholes. ``K = 0`` with ``option="call"`` is the underlying's own quote
    ``(S exp(m_bid theta T), S exp(m_ask theta T))``.
    """
    mb, ma = shift(kind, option)
    f = bs_call if option == "call" else bs_put
    S0 = np.asarray(S0, float)
    return (f(S0 * np.exp(mb * theta * T), K, r, sigma, T), f(S0 * np.exp(ma * theta * T), K, r, sigma, T))


def stock_quote(S0: float, theta: float, T: float = T_YEARS, kind: Kind = "symmetric") -> Tuple[float, float]:
    """Model bid/ask of the underlying itself (a strike-0 call)."""
    b, a = ds_gbm_bid_ask(S0, 0.0, "call", 0.3, theta, T, R_CONT, kind)
    return float(b), float(a)


def model_quotes(q: Quotes, sigma: float, theta: float, kind: Kind = "symmetric", r: float = R_CONT,
                 T: float = T_YEARS, floor: float = 0.0):
    """Model ``(call_bid, call_ask, put_bid, put_ask)`` at the quoted strikes, rooted at ``q.S0_bid`` as in the tree code.

    ``floor`` adds an additive ``+/- floor/2`` (bid clipped at 0): a *diagnostic* friction term, not part of the DS limit.
    """
    cb, ca = ds_gbm_bid_ask(q.S0_bid, q.K_call, "call", sigma, theta, T, r, kind)
    pb, pa = ds_gbm_bid_ask(q.S0_bid, q.K_put, "put", sigma, theta, T, r, kind)
    return (np.maximum(cb - floor / 2, 0.0), ca + floor / 2, np.maximum(pb - floor / 2, 0.0), pa + floor / 2)


def pde_bid_ask(payoff: Payoff, S0: float, sigma: float, theta: float, T: float = T_YEARS, r: float = R_CONT,
                kind: Kind = "symmetric", side: Literal["bid", "ask"] = "ask", dx: float = 0.002,
                half: float = 0.9) -> float:
    """Value of any payoff from the limit PDE (Theorem 2), by an explicit monotone upwind scheme in ``x = ln S``.

    Each control ``s in {-1, +1}`` (drift ``r + m_c - sigma^2/2 + s theta``) is upwinded separately and the max (ask) or
    min (bid) is taken, so the scheme is monotone, stable (``dt = 0.4 dx^2 / sigma^2``) and consistent. First-order
    accurate in ``dx``; the domain is ``ln S0 +/- half`` with linear extrapolation at the boundary.
    """
    m_c = theta if kind == "thesis" else 0.0
    drift0 = r + m_c - 0.5 * sigma**2
    x0 = float(np.log(S0))
    x = np.arange(x0 - half, x0 + half + dx / 2, dx)
    i0 = int(round((x0 - x[0]) / dx))
    v = np.asarray(payoff(np.exp(x)), float)
    dt = 0.4 * dx**2 / sigma**2
    ns = int(np.ceil(T / dt)); dt = T / ns
    for _ in range(ns):
        vxx = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx**2
        best = None
        for s in (-1.0, 1.0):
            b = drift0 + s * theta
            term = b * ((v[2:] - v[1:-1]) / dx if b > 0 else (v[1:-1] - v[:-2]) / dx)
            best = term if best is None else (np.maximum(best, term) if side == "ask" else np.minimum(best, term))
        new = v.copy()
        new[1:-1] = v[1:-1] + dt * (0.5 * sigma**2 * vxx + best - r * v[1:-1])
        new[0] = 2 * new[1] - new[2]; new[-1] = 2 * new[-2] - new[-3]
        v = new
    return float(v[i0])


def _loss(x, kind, q, r, T, with_floor, drop_stock):
    cb, ca, pb, pa = model_quotes(q, x[0], x[1], kind, r, T, x[2] if with_floor else 0.0)
    keep = (q.K_call > 0) if drop_stock else np.ones(len(q.K_call), bool)
    return float(np.sum(((cb - q.C_bid) ** 2 + (ca - q.C_ask) ** 2)[keep]) + np.sum((pb - q.P_bid) ** 2 + (pa - q.P_ask) ** 2))


def calibrate_continuous(q: Quotes, kind: Kind = "symmetric", r: float = R_CONT, T: float = T_YEARS,
                         with_floor: bool = False, drop_stock: bool = False, seeds: Sequence[int] = (0, 1, 2),
                         sigma_bounds: Tuple[float, float] = (0.05, 1.5), theta_bounds: Tuple[float, float] = (0.0, 3.0),
                         floor_max: float = 0.6, maxiter: int = 400) -> Dict[str, float]:
    """Fit ``(sigma, theta[, floor])`` to the thesis price loss (eq. 30) with differential evolution (best of ``seeds``).

    ``drop_stock`` removes the strike-0 underlying quote from the loss, to test what the *options alone* imply for the
    underlying's spread. Flat ``sigma`` means no smile: see ``docs/CONTINUOUS_LIMIT.md`` for what that costs.
    """
    bounds = [sigma_bounds, theta_bounds] + ([(0.0, floor_max)] if with_floor else [])
    best = None
    for s in seeds:
        res = differential_evolution(_loss, bounds, args=(kind, q, r, T, with_floor, drop_stock), seed=s, tol=1e-12,
                                     maxiter=maxiter, popsize=25, polish=True)
        if best is None or res.fun < best.fun:
            best = res
    out = dict(kind=kind, sigma=float(best.x[0]), theta=float(best.x[1]), error=float(best.fun))
    if with_floor:
        out["floor"] = float(best.x[2])
    return out
