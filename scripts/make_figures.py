"""Regenerate all thesis figures into figures/: python scripts/make_figures.py"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dsbidask import *
from dsbidask.data import load_quotes
from dsbidask.theory import spread_vs_epsilon
from dsbidask.pricing import european_tree
from dsbidask import plotting as P

OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)
q = load_quotes()
P.plot_market_quotes(q.K_call, q.C_bid, q.C_ask, "META calls", OUT / "market_calls.png")
P.plot_market_quotes(q.K_put, q.P_bid, q.P_ask, "META puts", OUT / "market_puts.png")

# Theoretical spread vs epsilon (R=1.01, u=1.05, d=0.95, S0=K=100)
for kind in ("call", "put"):
    curves = {f"T={T}": spread_vs_epsilon(kind, 100, 100, 1.05, 0.95, 1.01, T) for T in (5, 10, 15, 20)}
    P.plot_spread_vs_epsilon(curves, "Ask - bid", f"European {kind}: theoretical spread vs $\\epsilon$",
                             OUT / f"theory_{kind}_spread.png")

# Calibrated fit (free d; parameters from the thesis run, seed 12345)
params = dict(u=1.02354592, d=0.96425885, b_d=0.39260853)
P.plot_calibration_fit(q, params, R_DEFAULT, T_DEFAULT, OUT / "calibration_fit.png")

# Bid/ask trees of a European call (toy example: S0=100, K=100, T=5)
S, bid, ask = european_tree(lambda s: np.maximum(s - 100, 0), 100, 1.2, 0.8, 0.4 * 0.999, 1.04, 5, "increasing")
P.plot_tree(S, "Stock tree", OUT / "tree_stock.png")
P.plot_tree(bid, "European call: bid tree", OUT / "tree_call_bid.png")
P.plot_tree(ask, "European call: ask tree", OUT / "tree_call_ask.png")
print("figures written to", OUT)
