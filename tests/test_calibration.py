import numpy as np
from dsbidask import R_DEFAULT
from dsbidask.calibration import (calibrate, calibrate_multistart, fit_report, objective, polish, sse)
from dsbidask.data import load_quotes

q = load_quotes()
P = dict(u=1.02354592, d=0.96425885, b_d=0.39260853)


def test_thesis_objective_value():
    assert abs(sse(P["u"], P["d"], P["b_d"], q) - 15.4418) < 1e-3


def test_price_objective_is_twice_mid_spread_objective_at_quarter_weight():
    a = objective(**P, q=q, mode="price")
    b = objective(**P, q=q, mode="mid_spread", spread_weight=0.25)
    assert abs(a - 2 * b) < 1e-9


def test_inv_spread_objective_is_finite_and_positive():
    assert 0 < objective(**P, q=q, mode="inv_spread") < np.inf


def test_fixed_d_reproduces_thesis_fit():
    r = calibrate(q, free_d=False, maxiter=2000, seed=12345)
    assert abs(r["error"] - 21.950091) < 1e-4 and abs(r["u"] - 1.029428) < 1e-5


def test_multistart_table_sorted_and_polish_never_worse():
    best, table = calibrate_multistart(q, free_d=True, seeds=range(4), maxiter=300)
    assert list(table.error) == sorted(table.error)
    assert (table.error <= table.error_pso + 1e-9).all()
    assert best["error"] == table.error.min()
    assert best["b_u"] + best["b_d"] <= 1 + 1e-9


def test_polish_respects_constraint_and_bounds():
    p = polish(q, [P["u"], P["d"], P["b_d"]], free_d=True)
    assert p["error"] <= sse(P["u"], P["d"], P["b_d"], q) + 1e-9
    assert 0.3 - 1e-9 <= p["b_d"] <= 0.6 + 1e-9 and p["b_u"] + p["b_d"] <= 1 + 1e-9


def test_fit_report_keys():
    r = fit_report(**P, q=q)
    assert set(r) == {"rmse_price", "mean_model_spread", "mean_market_spread"}
    assert r["mean_model_spread"] < r["mean_market_spread"]
