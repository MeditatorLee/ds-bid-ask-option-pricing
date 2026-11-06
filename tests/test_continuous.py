import numpy as np
import pytest

from dsbidask.ambiguity import Ambiguity, chain_bid_ask
from dsbidask.continuous import (R_CONT, bs_call, bs_put, calibrate_continuous, ds_gbm_bid_ask, model_quotes,
                                 pde_bid_ask, shift, stock_quote)
from dsbidask.data import Quotes

S0, SIG, TH, T, R = 300.11, 0.30, 0.30, 0.08, R_CONT
KS = np.array([270.0, 300.0, 330.0])


def tree(n, kind, payoff, S=S0):
    """Scaled DS tree (u = e^{sigma sqrt(D)}, eps = theta sqrt(D)/sigma) for an arbitrary payoff, 1-D recursion."""
    D = T / n; u = np.exp(SIG * np.sqrt(D)); d = 1 / u; Rn = np.exp(R * D); q = (Rn - d) / (u - d); eps = TH * np.sqrt(D) / SIG
    bu, bd = (q, 1 - q - eps) if kind == "thesis" else (q - eps / 2, 1 - q - eps / 2)
    sl = 1 - bu - bd
    bid = ask = payoff(S * u ** (2 * np.arange(n + 1) - n))
    for _ in range(n):
        ub, db, ua, da = bid[1:], bid[:-1], ask[1:], ask[:-1]
        bid = (bu * ub + bd * db + sl * np.minimum(ub, db)) / Rn
        ask = (bu * ua + bd * da + sl * np.maximum(ua, da)) / Rn
    return float(bid[0]), float(ask[0])


def test_black_scholes_reference_value_and_parity():
    assert abs(bs_call(100.0, 100.0, 0.05, 0.2, 1.0) - 10.4506) < 1e-3
    c, p = bs_call(S0, KS, R, SIG, T), bs_put(S0, KS, R, SIG, T)
    assert np.allclose(c - p, S0 - KS * np.exp(-R * T))
    assert bs_call(S0, 0.0, R, SIG, T) == pytest.approx(S0)


@pytest.mark.parametrize("kind", ["thesis", "symmetric"])
@pytest.mark.parametrize("option", ["call", "put"])
def test_zero_theta_is_black_scholes_with_zero_spread(kind, option):
    b, a = ds_gbm_bid_ask(S0, KS, option, SIG, 0.0, T, R, kind)
    ref = (bs_call if option == "call" else bs_put)(S0, KS, R, SIG, T)
    assert np.allclose(b, ref) and np.allclose(a, ref)


@pytest.mark.parametrize("kind", ["thesis", "symmetric"])
@pytest.mark.parametrize("option", ["call", "put"])
def test_closed_form_is_the_limit_of_the_scaled_tree(kind, option):
    n = 1000
    D = T / n; u = np.exp(SIG * np.sqrt(D)); eps = TH * np.sqrt(D) / SIG
    b, a = chain_bid_ask(S0, KS, option, u, 1 / u, np.exp(R * D), n, Ambiguity(eps, kind))
    cb, ca = ds_gbm_bid_ask(S0, KS, option, SIG, TH, T, R, kind)
    assert np.max(np.abs(b - cb)) < 5e-3 and np.max(np.abs(a - ca)) < 5e-3


def test_forward_interval_and_stock_quote():
    for kind in ("thesis", "symmetric"):
        mb, ma = shift(kind, "call")
        for K in KS:
            cb, ca = ds_gbm_bid_ask(S0, K, "call", SIG, TH, T, R, kind)
            pb, pa = ds_gbm_bid_ask(S0, K, "put", SIG, TH, T, R, kind)
            disc = K * np.exp(-R * T)
            assert cb - pa == pytest.approx(S0 * np.exp(mb * TH * T) - disc, abs=1e-9)   # call bid / put ask share a spot
            assert ca - pb == pytest.approx(S0 * np.exp(ma * TH * T) - disc, abs=1e-9)
        sb, sa = stock_quote(S0, TH, T, kind)
        assert (sb, sa) == pytest.approx((S0 * np.exp(mb * TH * T), S0 * np.exp(ma * TH * T)))
    sb, sa = stock_quote(S0, TH, T, "symmetric")
    assert sb == pytest.approx(S0 * np.exp(-TH * T)) and sa == pytest.approx(S0 * np.exp(TH * T))


@pytest.mark.parametrize("kind", ["thesis", "symmetric"])
def test_option_spread_never_exceeds_underlying_spread(kind):
    sb, sa = stock_quote(S0, TH, T, kind)
    K = np.linspace(1.0, 900.0, 400)
    for option in ("call", "put"):
        b, a = ds_gbm_bid_ask(S0, K, option, SIG, TH, T, R, kind)
        assert (a - b >= -1e-12).all() and (a - b <= (sa - sb) + 1e-9).all()


def test_spread_increases_with_theta():
    s = [np.subtract(*ds_gbm_bid_ask(S0, 300.0, "call", SIG, th, T, R, "symmetric")[::-1]) for th in (0.0, 0.1, 0.3, 0.6)]
    assert all(x < y for x, y in zip(s, s[1:]))


@pytest.mark.parametrize("kind", ["thesis", "symmetric"])
def test_pde_solver_matches_closed_form_for_a_call(kind):
    pay = lambda s: np.maximum(s - 300.0, 0.0)
    cb, ca = ds_gbm_bid_ask(S0, 300.0, "call", SIG, TH, T, R, kind)
    for side, ref in (("bid", cb), ("ask", ca)):
        coarse = abs(pde_bid_ask(pay, S0, SIG, TH, T, R, kind, side, dx=0.002) - ref)
        fine = abs(pde_bid_ask(pay, S0, SIG, TH, T, R, kind, side, dx=0.001) - ref)
        assert fine < 0.05 and fine < 0.7 * coarse        # first-order scheme: halving dx roughly halves the error


def test_pde_solver_matches_tree_for_a_straddle():
    pay = lambda s: np.abs(s - 300.0)
    tb, ta = tree(2000, "symmetric", pay)
    assert abs(pde_bid_ask(pay, S0, SIG, TH, T, R, "symmetric", "bid", dx=0.001) - tb) < 0.06
    assert abs(pde_bid_ask(pay, S0, SIG, TH, T, R, "symmetric", "ask", dx=0.001) - ta) < 0.06
    assert tb < bs_call(S0, 300.0, R, SIG, T) + bs_put(S0, 300.0, R, SIG, T) < ta


def test_calibration_recovers_synthetic_parameters():
    K = np.arange(250.0, 351.0, 10.0)
    sb, sa = stock_quote(S0, 0.02, T, "symmetric")
    cb, ca = ds_gbm_bid_ask(S0, K, "call", 0.4, 0.02, T, R, "symmetric")
    pb, pa = ds_gbm_bid_ask(S0, K, "put", 0.4, 0.02, T, R, "symmetric")
    q = Quotes(np.r_[0.0, K], np.r_[sb, cb], np.r_[sa, ca], K, pb, pa, S0_bid=S0, S0_ask=sa)
    cb2, *_ = model_quotes(q, 0.4, 0.02, "symmetric")
    assert np.allclose(cb2, np.r_[sb, cb])
    r = calibrate_continuous(q, "symmetric", seeds=(0,), maxiter=200)
    assert abs(r["sigma"] - 0.4) < 1e-3 and abs(r["theta"] - 0.02) < 1e-3 and r["error"] < 1e-6
