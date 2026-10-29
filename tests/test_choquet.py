import numpy as np
from dsbidask import *
from dsbidask.pricing import european_calls_bid_ask, european_puts_bid_ask

U, D, BD = 1.02354592, 0.96425885, 0.39260853
R, T = R_DEFAULT, 20


def test_recursion_matches_closed_form_for_monotone_payoffs():
    for K in (250, 300, 350):
        call = lambda s: np.maximum(s - K, 0.0); put = lambda s: np.maximum(K - s, 0.0)
        assert np.allclose(bid_ask(call, 300.11, U, D, BD, R, T), european_call_bid_ask(300.11, K, U, D, BD, R, T))
        assert np.allclose(bid_ask(put, 300.11, U, D, BD, R, T), european_put_bid_ask(300.11, K, U, D, BD, R, T))


def test_vectorised_matches_scalar():
    K = np.array([200., 300., 400.])
    b, a = european_calls_bid_ask(300.11, K, U, D, BD, R, T)
    assert np.allclose([european_call_bid_ask(300.11, k, U, D, BD, R, T) for k in K], np.c_[b, a])


def test_straddle_bid_below_ask_and_closed_form_is_invalid():
    S0, K, Tt, u, d, Rr, bd = 100, 100, 5, 1.2, 0.8, 1.04, 0.4 * 0.999
    pay = lambda s: np.abs(s - K)
    bid, ask = bid_ask(pay, S0, u, d, bd, Rr, Tt)
    assert bid < ask
    # monotone closed form misapplied to a straddle gives bid > ask (thesis appendix error)
    wb = european_bid_ask(pay, S0, u, d, bd, Rr, Tt, "decreasing")[0]
    wa = european_bid_ask(pay, S0, u, d, bd, Rr, Tt, "decreasing")[1]
    assert wb > wa


def test_zero_ambiguity_gives_equal_bid_ask_for_any_payoff():
    bd = 1 - b_up(1.2, 0.8, 1.04)
    b, a = bid_ask(lambda s: np.abs(s - 100), 100, 1.2, 0.8, bd, 1.04, 5)
    assert abs(b - a) < 1e-10


def test_american_geq_european_and_early_exercise_exists():
    pay = lambda s: np.maximum(350 - s, 0.0)
    ab, aa = american_put_bid_ask(300.11, 350, U, D, BD, R, T)
    eb, ea = bid_ask(pay, 300.11, U, D, BD, R, T)
    assert ab >= eb - 1e-12 and aa >= ea - 1e-12 and ab <= aa
    fb, fa = early_exercise_nodes(pay, 300.11, U, D, BD, R, T)
    assert fb.any()


def test_replication_prop_4_4():
    kw = dict(S0=100, u=1.2, d=0.8, b_d=0.4 * 0.999, R=1.04, T=5)
    call = lambda s: np.maximum(s - 100, 0.0); put = lambda s: np.maximum(100 - s, 0.0)
    S, cb, ca = bid_ask_trees(call, **kw)
    _, pb, pa = bid_ask_trees(put, **kw)
    assert is_self_financing(replicating_portfolio(S, cb, kw["R"])[2])        # call bid: lambda1 >= 0
    assert not is_self_financing(replicating_portfolio(S, ca, kw["R"])[2])    # call ask: not self-financing
    assert is_self_financing(replicating_portfolio(S, pa, kw["R"])[2])        # put ask: delta1 <= 0
    assert not is_self_financing(replicating_portfolio(S, pb, kw["R"])[2])    # put bid: not self-financing
