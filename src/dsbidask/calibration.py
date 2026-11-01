"""Calibration of ``(u, d, b_d)`` (free ``d``) or ``(u, b_d)`` with ``d = 1/u`` (thesis Chapter 5).

* :func:`sse` is the thesis objective, eq. (30), vectorised over strikes.
* :func:`objective` adds two spread-aware variants (not in the thesis), see its docstring.
* :func:`calibrate` is one PSO run (what the thesis did, with seed 12345).
* :func:`calibrate_multistart` runs PSO from many seeds and polishes each with SLSQP, because the
  free-``d`` landscape has many local minima (see ``docs/THESIS_ERRATA.md``, C5).
"""
from __future__ import annotations

import inspect
from typing import Callable, Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyswarm import pso
from scipy.optimize import minimize

from .config import R_DEFAULT, T_DEFAULT
from .data import Quotes
from .pricing import b_up, european_calls_bid_ask, european_puts_bid_ask

Mode = Literal["price", "inv_spread", "mid_spread"]


def model_quotes(u: float, d: float, b_d: float, q: Quotes, R: float = R_DEFAULT, T: int = T_DEFAULT):
    """Model ``(call_bid, call_ask, put_bid, put_ask)`` at the quoted strikes (Prop. 4.1-4.2)."""
    cb, ca = european_calls_bid_ask(q.S0_bid, q.K_call, u, d, b_d, R, T)
    pb, pa = european_puts_bid_ask(q.S0_bid, q.K_put, u, d, b_d, R, T)
    return cb, ca, pb, pa


def objective(u: float, d: float, b_d: float, q: Quotes, R: float = R_DEFAULT, T: int = T_DEFAULT,
              mode: Mode = "price", spread_weight: float = 0.25, spread_floor: float = 0.01) -> float:
    """Calibration error ``E(u, d, b_d)``.

    * ``"price"``: thesis eq. (30), ``sum (bid err)^2 + (ask err)^2`` over all calls and puts.
    * ``"mid_spread"``: ``sum (mid err)^2 + spread_weight * (spread err)^2``. Because
      ``e_bid^2 + e_ask^2 = 2 e_mid^2 + e_spread^2 / 2``, the thesis objective equals ``2 *`` this one with
      ``spread_weight = 0.25``; larger weights emphasise reproducing the market spreads.
    * ``"inv_spread"``: each quote's bid and ask errors divided by ``max(market spread, spread_floor)``,
      so tightly quoted options must be fitted more precisely.
    """
    cb, ca, pb, pa = model_quotes(u, d, b_d, q, R, T)
    mb = np.r_[q.C_bid, q.P_bid]; ma = np.r_[q.C_ask, q.P_ask]
    xb = np.r_[cb, pb]; xa = np.r_[ca, pa]
    if mode == "price":
        return float(np.sum((xb - mb) ** 2) + np.sum((xa - ma) ** 2))
    if mode == "mid_spread":
        e_mid = (xb + xa) / 2 - (mb + ma) / 2
        e_spr = (xa - xb) - (ma - mb)
        return float(np.sum(e_mid**2) + spread_weight * np.sum(e_spr**2))
    if mode == "inv_spread":
        w = 1.0 / np.maximum(ma - mb, spread_floor)
        return float(np.sum(((xb - mb) * w) ** 2) + np.sum(((xa - ma) * w) ** 2))
    raise ValueError(f"unknown mode {mode!r}")


def sse(u: float, d: float, b_d: float, q: Quotes, R: float = R_DEFAULT, T: int = T_DEFAULT) -> float:
    """Thesis squared error, eq. (30)."""
    return objective(u, d, b_d, q, R, T, mode="price")


def fit_report(u: float, d: float, b_d: float, q: Quotes, R: float = R_DEFAULT, T: int = T_DEFAULT) -> Dict[str, float]:
    """Price RMSE and mean model/market spreads (excluding the strike-0 stock quote)."""
    cb, ca, pb, pa = model_quotes(u, d, b_d, q, R, T)
    mb = np.r_[q.C_bid, q.P_bid]; ma = np.r_[q.C_ask, q.P_ask]
    xb = np.r_[cb, pb]; xa = np.r_[ca, pa]
    keep = np.r_[q.K_call > 0, np.ones(len(q.K_put), bool)]
    return dict(
        rmse_price=float(np.sqrt(np.mean(np.r_[(xb - mb) ** 2, (xa - ma) ** 2]))),
        mean_model_spread=float(np.mean((xa - xb)[keep])),
        mean_market_spread=float(np.mean((ma - mb)[keep])),
    )


def _problem(q: Quotes, free_d: bool, R: float, T: int, u_max: float, bd_bounds: Tuple[float, float],
             eps: float, **obj_kw):
    """Objective, constraint (``b_u + b_d <= 1``) and box bounds in the thesis parametrisation."""
    if free_d:
        f = lambda x: objective(x[0], x[1], x[2], q, R, T, **obj_kw)
        con = lambda x: [1 - b_up(x[0], x[1], R) - x[2]]
        lb, ub = [R + eps, 0.0, bd_bounds[0]], [u_max, R - eps, bd_bounds[1]]
    else:
        f = lambda x: objective(x[0], 1 / x[0], x[1], q, R, T, **obj_kw)
        con = lambda x: [1 - b_up(x[0], 1 / x[0], R) - x[1]]
        lb, ub = [R + eps, bd_bounds[0]], [u_max, bd_bounds[1]]
    return f, con, lb, ub


def _run_pso(f: Callable, lb, ub, con: Callable, swarmsize: int, maxiter: int, seed: int):
    """Call ``pyswarm.pso`` across API versions (<=0.6 returns a tuple; >=1.0 an OptimizeResult).

    Results are only bit-for-bit reproducible within one pyswarm version: the thesis used 0.6.
    """
    kw = dict(f_ieqcons=lambda x: np.asarray(con(x), float), swarmsize=swarmsize, maxiter=maxiter, debug=False)
    if "seed" in inspect.signature(pso).parameters:
        res = pso(f, lb, ub, seed=seed, **kw)
    else:
        np.random.seed(seed)
        res = pso(f, lb, ub, **kw)
    return (res.x, res.fun) if hasattr(res, "x") else res


def _pack(x, err: float, free_d: bool, R: float) -> Dict[str, float]:
    u, d, b_d = (x[0], x[1], x[2]) if free_d else (x[0], 1 / x[0], x[1])
    return dict(u=float(u), d=float(d), b_d=float(b_d), b_u=float(b_up(u, d, R)), error=float(err))


def calibrate(q: Quotes, free_d: bool = True, R: float = R_DEFAULT, T: int = T_DEFAULT, swarmsize: int = 100,
              maxiter: int = 2000, seed: int = 12345, u_max: float = 1.056,
              bd_bounds: Tuple[float, float] = (0.3, 0.6), eps: float = 1e-5, **obj_kw) -> Dict[str, float]:
    """One PSO run (eq. 31). Box bounds follow the thesis *code*: ``u in (R, u_max]``, ``b_d in bd_bounds``.

    Note: the thesis states problem (31) without these box bounds. Extra keywords (``mode``,
    ``spread_weight``, ``spread_floor``) select the objective, see :func:`objective`.
    """
    f, con, lb, ub = _problem(q, free_d, R, T, u_max, bd_bounds, eps, **obj_kw)
    x, err = _run_pso(f, lb, ub, con, swarmsize, maxiter, seed)
    return _pack(x, err, free_d, R)


def polish(q: Quotes, x0: Sequence[float], free_d: bool = True, R: float = R_DEFAULT, T: int = T_DEFAULT,
           u_max: float = 1.056, bd_bounds: Tuple[float, float] = (0.3, 0.6), eps: float = 1e-5,
           **obj_kw) -> Dict[str, float]:
    """Local refinement of a PSO point with SLSQP (same bounds and ``b_u + b_d <= 1`` constraint)."""
    f, con, lb, ub = _problem(q, free_d, R, T, u_max, bd_bounds, eps, **obj_kw)
    res = minimize(f, np.asarray(x0, float), method="SLSQP", bounds=list(zip(lb, ub)),
                   constraints=[{"type": "ineq", "fun": lambda x: np.asarray(con(x))}],
                   options=dict(maxiter=500, ftol=1e-12))
    x = res.x if (res.success or res.fun < f(x0)) else np.asarray(x0, float)
    ok = con(x)[0] >= -1e-10 and np.all(x >= np.array(lb) - 1e-12) and np.all(x <= np.array(ub) + 1e-12)
    if not ok or f(x) > f(x0):
        x = np.asarray(x0, float)
    return _pack(x, f(x), free_d, R)


def calibrate_multistart(q: Quotes, free_d: bool = True, seeds: Iterable[int] = range(50), do_polish: bool = True,
                         **kw) -> Tuple[Dict[str, float], pd.DataFrame]:
    """PSO from many seeds, each optionally polished by SLSQP.

    Returns ``(best, table)`` where ``table`` has one row per seed with the PSO and polished errors,
    so the dispersion of local optima can be reported.
    """
    obj_keys = ("mode", "spread_weight", "spread_floor")
    obj_kw = {k: kw[k] for k in obj_keys if k in kw}
    pso_kw = {k: v for k, v in kw.items() if k not in obj_keys}
    rows = []
    for s in seeds:
        r = calibrate(q, free_d, seed=s, **pso_kw, **obj_kw)
        row = dict(seed=s, **{k: r[k] for k in ("u", "d", "b_d", "b_u")}, error_pso=r["error"])
        if do_polish:
            x0 = [r["u"], r["d"], r["b_d"]] if free_d else [r["u"], r["b_d"]]
            p = polish(q, x0, free_d, **{k: v for k, v in pso_kw.items() if k in ("R", "T", "u_max", "bd_bounds", "eps")},
                       **obj_kw)
            row.update({k: p[k] for k in ("u", "d", "b_d", "b_u")}); row["error"] = p["error"]
        else:
            row["error"] = r["error"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("error").reset_index(drop=True)
    best = table.iloc[0].to_dict()
    return best, table
