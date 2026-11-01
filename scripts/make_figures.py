"""Regenerate all thesis figures into figures/: python scripts/make_figures.py"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import argparse
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
from dsbidask import *
from dsbidask.data import load_quotes
from dsbidask.theory import spread_vs_epsilon
from dsbidask.pricing import european_tree
from dsbidask.theory import put_spread_vs_strike
from dsbidask.choquet import bid_ask_trees, early_exercise_nodes
from dsbidask.replication import replicating_portfolio
from dsbidask import plotting as P

ap = argparse.ArgumentParser()
ap.add_argument("--dpi", type=int, default=150, help="figure resolution (default 150; thesis-quality: 300)")
args = ap.parse_args()
P.DPI = args.dpi
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
# Fig. 7 analogue: calibrated put spread vs strike, European vs American (S0 = 300.11)
Ks = np.arange(200, 400.25, 0.25)
sp = {}
for am in (False, True):
    _, _, s_ = put_spread_vs_strike(S0_BID, Ks, params["u"], params["d"], params["b_d"], R_DEFAULT, T_DEFAULT, american=am)
    sp["American" if am else "European"] = s_
P.plot_put_spreads_vs_strike(Ks, sp, path=OUT / "put_spread_vs_strike.png")

# Corrected thesis Fig. 3: straddle trees by backward recursion (S0=K=100, T=5, u=1.2, d=0.8, r=0.04)
kw = dict(S0=100, u=1.2, d=0.8, b_d=0.4 * 0.999, R=1.04, T=5)
S, sb, sa = bid_ask_trees(lambda s: np.abs(s - 100), **kw)
P.plot_tree(sb, "Straddle: bid tree (recursion)", OUT / "tree_straddle_bid.png")
P.plot_tree(sa, "Straddle: ask tree (recursion)", OUT / "tree_straddle_ask.png")

# American put early-exercise nodes (K=350 on the META tree)
pay = lambda s: np.maximum(350 - s, 0.0)
St, ab, aa = bid_ask_trees(pay, S0_BID, params["u"], params["d"], params["b_d"], R_DEFAULT, 8, american=True)
fb, fa = early_exercise_nodes(pay, S0_BID, params["u"], params["d"], params["b_d"], R_DEFAULT, 8)
P.plot_tree(ab, "American put bid (red = exercise), K=350, T=8", OUT / "tree_american_put_bid.png", highlight=fb)

# Replication (Prop. 4.4): call bid strategy is self-financing, call ask is not
S, cb, ca = bid_ask_trees(lambda s: np.maximum(s - 100, 0.0), **kw)
for lab, V in (("bid", cb), ("ask", ca)):
    l0, l1, res = replicating_portfolio(S, V, kw["R"])
    P.plot_replication(S, V, l0, l1, res, f"Call {lab}", OUT / f"replication_call_{lab}.png")
# Calibration robustness figures (from results/*.csv produced by the experiments; skipped if absent)
import pandas as pd
RES = ROOT / "results"
tabs = {n: pd.read_csv(RES / f"multistart_{k}.csv") for n, k in
        (("free d, thesis bounds", "free_d_thesis_bounds"), ("free d, wide bounds", "free_d_wide_bounds"),
         ("d = 1/u", "fixed_d_thesis_bounds")) if (RES / f"multistart_{k}.csv").exists()}
if tabs:
    P.plot_multistart_errors(tabs, thesis_value=15.4418, path=OUT / "multistart_errors.png")
if (RES / "objective_comparison.csv").exists():
    P.plot_objective_tradeoff(pd.read_csv(RES / "objective_comparison.csv"), path=OUT / "objective_tradeoff.png")
print("figures written to", OUT)
