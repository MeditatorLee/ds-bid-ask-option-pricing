"""Properties of the American extension (not a thesis result; see docs/AMERICAN.md)."""
import numpy as np
import pytest
from dsbidask import *
from dsbidask.choquet import bid_ask_trees

U, D, BD = 1.02354592, 0.96425885, 0.39260853
R, T, S0 = R_DEFAULT, 20, 300.11


def original_authors_algorithm(S0, K, u, d, b_d, R, T):
    """Port of AmericanPut.py (fixed weights; j = number of DOWN moves in the original indexing)."""
    b_u = (R - d) / (u - d)
    S = np.zeros((T + 1, T + 1)); S[0, 0] = S0; z = 1
    for t in range(1, T + 1):
        for j in range(z):
            S[j, t] = S[j, t - 1] * u; S[j + 1, t] = S[j, t - 1] * d
        z += 1
    bid = np.zeros_like(S); ask = np.zeros_like(S)
    bid[:, T] = ask[:, T] = np.maximum(K - S[:, T], 0)
    ex_b = np.zeros_like(S); ex_a = np.zeros_like(S); z = T
    for t in range(T - 1, -1, -1):
        for j in range(z):
            e = max(K - S[j, t], 0)
            pb = ((1 - b_d) * bid[j, t + 1] + b_d * bid[j + 1, t + 1]) / R
            pa = (b_u * ask[j, t + 1] + (1 - b_u) * ask[j + 1, t + 1]) / R
            bid[j, t], ask[j, t] = max(pb, e), max(pa, e)
            ex_b[j, t], ex_a[j, t] = e > pb, e > pa
        z -= 1
    return bid[0, 0], ask[0, 0], ex_b.sum(), ex_a.sum()


@pytest.mark.parametrize("K", [250.0, 300.0, 350.0, 400.0])
def test_matches_original_authors_code(K):
    ob, oa, nb, na = original_authors_algorithm(S0, K, U, D, BD, R, T)
    ab, aa = american_put_bid_ask(S0, K, U, D, BD, R, T)
    assert np.allclose([ab, aa], [ob, oa], atol=1e-10)
    fb, fa = early_exercise_nodes(lambda s: np.maximum(K - s, 0), S0, U, D, BD, R, T)
    assert (fb.sum(), fa.sum()) == (nb, na)


def test_put_continuation_is_monotone_so_fixed_weights_are_exact():
    """Why the original fixed-weight recursion equals the general Choquet one for puts."""
    for K in (250.0, 350.0):
        _, bid, ask = bid_ask_trees(lambda s: np.maximum(K - s, 0), S0, U, D, BD, R, T, american=True)
        for tree in (bid, ask):
            for n in range(T + 1):      # j = number of up moves: value must be non-increasing in j
                row = tree[n, : n + 1]
                assert np.all(np.diff(row) <= 1e-12)


@pytest.mark.parametrize("K", [250.0, 300.0, 350.0, 400.0])
def test_basic_properties(K):
    pay = lambda s: np.maximum(K - s, 0)
    ab, aa = american_put_bid_ask(S0, K, U, D, BD, R, T)
    eb, ea = bid_ask(pay, S0, U, D, BD, R, T)
    assert ab <= aa + 1e-12                      # bid <= ask
    assert ab >= eb - 1e-12 and aa >= ea - 1e-12  # American >= European
    assert ab >= max(K - S0, 0) - 1e-12          # >= intrinsic value


def test_monotone_in_strike():
    prev = (-1.0, -1.0)
    for K in np.arange(200, 420, 20.0):
        cur = american_put_bid_ask(S0, K, U, D, BD, R, T)
        assert cur[0] >= prev[0] - 1e-12 and cur[1] >= prev[1] - 1e-12
        prev = cur


def test_zero_ambiguity_equals_standard_crr_american_put():
    bd = 1 - b_up(U, D, R)
    p = b_up(U, D, R)
    K = 320.0
    V = np.maximum(K - S0 * U ** np.arange(T + 1) * D ** (T - np.arange(T + 1)), 0)
    for n in range(T - 1, -1, -1):          # independent textbook CRR American recursion
        S = S0 * U ** np.arange(n + 1) * D ** (n - np.arange(n + 1))
        V = np.maximum((p * V[1:] + (1 - p) * V[:-1]) / R, np.maximum(K - S, 0))
    b, a = american_put_bid_ask(S0, K, U, D, bd, R, T)
    assert abs(b - V[0]) < 1e-9 and abs(a - V[0]) < 1e-9


def test_american_call_without_dividends_equals_european():
    """Holds for the CRR case; checked here for the ambiguous case as well."""
    K = 280.0
    pay = lambda s: np.maximum(s - K, 0)
    eb, ea = bid_ask(pay, S0, U, D, BD, R, T)
    ab, aa = bid_ask(pay, S0, U, D, BD, R, T, american=True)
    assert abs(ab - eb) < 1e-9 and abs(aa - ea) < 1e-9
