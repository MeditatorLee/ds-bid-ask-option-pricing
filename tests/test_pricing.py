import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dsbidask import *
from dsbidask.data import load_quotes
from dsbidask.calibration import sse

U, D, BD = 1.02354592, 0.96425885, 0.39260853


def test_reproduces_thesis_error():
    assert abs(sse(U, D, BD, load_quotes()) - 15.4418) < 1e-3

def test_b_up():
    assert abs(b_up(U, D, R_DEFAULT) - 0.60649) < 1e-4

def test_bid_le_ask():
    for K in (250, 300, 350):
        b, a = european_call_bid_ask(300, K, U, D, BD, R_DEFAULT, 20); assert b <= a
        b, a = european_put_bid_ask(300, K, U, D, BD, R_DEFAULT, 20); assert b <= a

def test_collapses_to_crr_when_bu_plus_bd_is_one():
    bd = 1 - b_up(U, D, R_DEFAULT)
    b, a = european_call_bid_ask(300, 300, U, D, bd, R_DEFAULT, 20)
    crr = crr_price(lambda s: np.maximum(s - 300, 0), 300, U, D, R_DEFAULT, 20)
    assert abs(b - crr) < 1e-10 and abs(a - crr) < 1e-10

def test_american_put_geq_european():
    for K in (280, 300, 320):
        ab, aa = american_put_bid_ask(300, K, U, D, BD, R_DEFAULT, 20)
        eb, ea = european_put_bid_ask(300, K, U, D, BD, R_DEFAULT, 20)
        assert ab >= eb - 1e-12 and aa >= ea - 1e-12 and ab <= aa


def test_european_tree_root_matches_closed_form():
    from dsbidask.pricing import european_tree
    S, bid, ask = european_tree(lambda s: np.maximum(s - 100, 0), 100, 1.2, 0.8, 0.399, 1.04, 5, "increasing")
    b, a = european_call_bid_ask(100, 100, 1.2, 0.8, 0.399, 1.04, 5)
    assert abs(bid[0, 0] - b) < 1e-12 and abs(ask[0, 0] - a) < 1e-12
    m = ~np.isnan(bid)
    assert np.all(bid[m] <= ask[m] + 1e-12)


def test_spread_vanishes_at_epsilon_one():
    from dsbidask.theory import spread_vs_epsilon
    e, s = spread_vs_epsilon("call", 100, 100, 1.05, 0.95, 1.01, 10, eps=[0.0, 0.5, 1.0])
    assert s[0] > s[1] > 0 and abs(s[2]) < 1e-10
