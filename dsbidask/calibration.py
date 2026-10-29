"""PSO calibration of (u, d, b_d) (free d) or (u, b_d) with d = 1/u."""
from __future__ import annotations
import numpy as np
from pyswarm import pso
from .config import R_DEFAULT, T_DEFAULT
from .pricing import european_call_bid_ask, european_put_bid_ask, b_up


def sse(u, d, b_d, q, R=R_DEFAULT, T=T_DEFAULT):
    e = 0.0
    for K, cb, ca in zip(q.K_call, q.C_bid, q.C_ask):
        b, a = european_call_bid_ask(q.S0_bid, K, u, d, b_d, R, T)
        e += (b - cb) ** 2 + (a - ca) ** 2
    for K, pb, pa in zip(q.K_put, q.P_bid, q.P_ask):
        b, a = european_put_bid_ask(q.S0_bid, K, u, d, b_d, R, T)
        e += (b - pb) ** 2 + (a - pa) ** 2
    return e


def calibrate(q, free_d=True, R=R_DEFAULT, T=T_DEFAULT, swarmsize=100, maxiter=2000, seed=12345):
    np.random.seed(seed)
    eps = 1e-5
    if free_d:
        f = lambda x: sse(x[0], x[1], x[2], q, R, T)
        con = lambda x: [1 - (R - x[1]) / (x[0] - x[1]) - x[2]]   # b_u + b_d <= 1
        lb, ub = [R + eps, 0.0, 0.3], [1.056, R - eps, 0.6]
    else:
        f = lambda x: sse(x[0], 1 / x[0], x[1], q, R, T)
        con = lambda x: [1 - (R - 1 / x[0]) / (x[0] - 1 / x[0]) - x[1]]
        lb, ub = [R + eps, 0.3], [1.056, 0.6]
    x, err = pso(f, lb, ub, f_ieqcons=con, swarmsize=swarmsize, maxiter=maxiter, debug=False)
    u, d, b_d = (x[0], x[1], x[2]) if free_d else (x[0], 1 / x[0], x[1])
    return dict(u=u, d=d, b_d=b_d, b_u=b_up(u, d, R), error=err)
