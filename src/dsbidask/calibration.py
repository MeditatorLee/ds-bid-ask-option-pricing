"""PSO calibration of (u, d, b_d) (free d) or (u, b_d) with d = 1/u  (thesis Chapter 5).

The squared-error objective (eq. 30) is fully vectorised over strikes.
"""
from __future__ import annotations
import inspect
import numpy as np
from pyswarm import pso
from .config import R_DEFAULT, T_DEFAULT
from .pricing import b_up, european_calls_bid_ask, european_puts_bid_ask


def sse(u, d, b_d, q, R=R_DEFAULT, T=T_DEFAULT):
    """Model squared error E(u, d, b_d) over all quoted calls and puts (bid and ask)."""
    cb, ca = european_calls_bid_ask(q.S0_bid, q.K_call, u, d, b_d, R, T)
    pb, pa = european_puts_bid_ask(q.S0_bid, q.K_put, u, d, b_d, R, T)
    return float(np.sum((cb - q.C_bid) ** 2) + np.sum((ca - q.C_ask) ** 2)
                 + np.sum((pb - q.P_bid) ** 2) + np.sum((pa - q.P_ask) ** 2))


def _run_pso(f, lb, ub, con, swarmsize, maxiter, seed):
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


def calibrate(q, free_d=True, R=R_DEFAULT, T=T_DEFAULT, swarmsize=100, maxiter=2000, seed=12345,
              u_max=1.056, bd_bounds=(0.3, 0.6), eps=1e-5):
    """Run PSO. Box bounds follow the thesis code: u in (R, u_max], b_d in bd_bounds.

    Note: the thesis states problem (31) without these box bounds; they are what the code uses.
    """
    if free_d:
        f = lambda x: sse(x[0], x[1], x[2], q, R, T)
        con = lambda x: [1 - b_up(x[0], x[1], R) - x[2]]            # b_u + b_d <= 1
        lb, ub = [R + eps, 0.0, bd_bounds[0]], [u_max, R - eps, bd_bounds[1]]
    else:
        f = lambda x: sse(x[0], 1 / x[0], x[1], q, R, T)
        con = lambda x: [1 - b_up(x[0], 1 / x[0], R) - x[1]]
        lb, ub = [R + eps, bd_bounds[0]], [u_max, bd_bounds[1]]
    x, err = _run_pso(f, lb, ub, con, swarmsize, maxiter, seed)
    u, d, b_d = (x[0], x[1], x[2]) if free_d else (x[0], 1 / x[0], x[1])
    return dict(u=float(u), d=float(d), b_d=float(b_d), b_u=float(b_up(u, d, R)), error=float(err))
