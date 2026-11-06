import numpy as np
import pytest

from dsbidask import R_DEFAULT as R, american_put_bid_ask, b_up, crr_price, european_calls_bid_ask, european_puts_bid_ask
from dsbidask.ambiguity import (Ambiguity, calibrate_ambiguity, chain_bid_ask, interval_report, loss, max_ambiguity,
                                model_quotes)
from dsbidask.data import load_quotes

U, D, BD, S0, T = 1.02354592, 0.96425885, 0.39260853, 300.11, 20
K = np.array([250.0, 290.0, 300.0, 310.0, 350.0])
QU = b_up(U, D, R)


def test_thesis_kind_constant_a_equals_closed_forms():
    amb = Ambiguity(a=1 - QU - BD, kind="thesis")
    cb, ca = chain_bid_ask(S0, K, "call", U, D, R, T, amb)
    pb, pa = chain_bid_ask(S0, K, "put", U, D, R, T, amb)
    ecb, eca = european_calls_bid_ask(S0, K, U, D, BD, R, T)
    epb, epa = european_puts_bid_ask(S0, K, U, D, BD, R, T)
    for x, y in ((cb, ecb), (ca, eca), (pb, epb), (pa, epa)):
        assert np.allclose(x, y, atol=1e-9)


@pytest.mark.parametrize("kind", ["thesis", "symmetric"])
def test_zero_ambiguity_is_crr_with_zero_spread(kind):
    amb = Ambiguity(a=0.0, kind=kind)
    for opt, sign in (("call", 1), ("put", -1)):
        bid, ask = chain_bid_ask(S0, K, opt, U, D, R, T, amb)
        ref = [crr_price(lambda s, k=k: np.maximum(sign * (s - k), 0), S0, U, D, R, T) for k in K]
        assert np.allclose(bid, ref, atol=1e-9) and np.allclose(ask, ref, atol=1e-9)


@pytest.mark.parametrize("kind", ["thesis", "symmetric"])
@pytest.mark.parametrize("width", [np.inf, 0.05, 0.2])
def test_bid_not_above_ask_and_prices_nonnegative(kind, width):
    amb = Ambiguity(a=0.5 * max_ambiguity(U, D, R, kind), kind=kind, width=width)
    for opt in ("call", "put"):
        bid, ask = chain_bid_ask(S0, K, opt, U, D, R, T, amb)
        assert (bid <= ask + 1e-12).all() and (bid >= -1e-12).all()


@pytest.mark.parametrize("kind", ["thesis", "symmetric"])
def test_state_dependent_belief_sets_are_nested_in_constant_ones(kind):
    """Less ambiguity at some nodes (same peak a) can only narrow [bid, ask]."""
    a = 0.5 * max_ambiguity(U, D, R, kind)
    const, bump = Ambiguity(a, kind), Ambiguity(a, kind, width=0.1)
    for opt in ("call", "put"):
        b0, a0 = chain_bid_ask(S0, K, opt, U, D, R, T, const)
        b1, a1 = chain_bid_ask(S0, K, opt, U, D, R, T, bump)
        assert (b1 >= b0 - 1e-12).all() and (a1 <= a0 + 1e-12).all()


def test_huge_width_equals_constant():
    a = 0.3 * max_ambiguity(U, D, R, "symmetric")
    x = chain_bid_ask(S0, K, "call", U, D, R, T, Ambiguity(a, "symmetric", width=1e6))
    y = chain_bid_ask(S0, K, "call", U, D, R, T, Ambiguity(a, "symmetric"))
    assert np.allclose(x, y, atol=1e-9)


def test_infeasible_ambiguity_raises():
    with pytest.raises(ValueError):
        chain_bid_ask(S0, K, "call", U, D, R, T, Ambiguity(a=max_ambiguity(U, D, R, "symmetric") + 0.01, kind="symmetric"))
    with pytest.raises(ValueError):
        chain_bid_ask(S0, K, "call", U, D, R, T, Ambiguity(a=-0.01, kind="thesis"))


def test_american_put_matches_existing_recursion_for_thesis_kind():
    amb = Ambiguity(a=1 - QU - BD, kind="thesis")
    for k in (250.0, 300.0, 350.0):
        bid, ask = chain_bid_ask(S0, k, "put", U, D, R, T, amb, american=True)
        eb, ea = american_put_bid_ask(S0, k, U, D, BD, R, T)
        assert abs(bid[0] - eb) < 1e-9 and abs(ask[0] - ea) < 1e-9


def test_symmetric_forward_interval_is_centred_at_one_step():
    """T = 1: (C_bid - P_ask) + (C_ask - P_bid) = 2 (S0 - K/R) for the symmetric kind, but not for the thesis kind."""
    k = np.array([300.0])  # strictly between S0*d and S0*u
    centred = {}
    for kind in ("symmetric", "thesis"):
        amb = Ambiguity(a=0.5 * max_ambiguity(U, D, R, kind), kind=kind)
        cb, ca = chain_bid_ask(S0, k, "call", U, D, R, 1, amb)
        pb, pa = chain_bid_ask(S0, k, "put", U, D, R, 1, amb)
        centred[kind] = float((cb - pa + ca - pb)[0] - 2 * (S0 - k[0] / R))
    assert abs(centred["symmetric"]) < 1e-9 and abs(centred["thesis"]) > 1e-3


def test_model_quotes_and_report_shapes():
    q = load_quotes()
    amb = Ambiguity(a=0.5 * max_ambiguity(U, D, R, "symmetric"), kind="symmetric")
    cb, ca, pb, pa = model_quotes(U, D, amb, q)
    assert len(cb) == len(q.K_call) and len(pa) == len(q.K_put)
    rep = interval_report(U, D, amb, q)
    assert set(rep) == {"rmse_price", "mean_model_spread", "mean_market_spread", "inside", "overlap", "mid_in"}
    assert np.isfinite(loss(U, D, amb, q)) and np.isfinite(loss(U, D, amb, q, mode="mid_spread", spread_weight=4.0))


def test_calibration_result_is_feasible_and_self_consistent():
    q = load_quotes()
    r = calibrate_ambiguity(q, "symmetric", state_dependent=True, seeds=(0,), maxiter=40, popsize=8)
    assert 0 <= r["a"] <= max_ambiguity(r["u"], r["d"], R, "symmetric") + 1e-12
    assert r["b_u"] >= -1e-12 and r["b_d"] >= -1e-12 and r["b_u"] + r["b_d"] <= 1 + 1e-12
    assert 0.01 <= r["width"] <= 0.5
    amb = Ambiguity(r["a"], "symmetric", r["width"])
    assert abs(r["error"] - loss(r["u"], r["d"], amb, q)) < 1e-9
