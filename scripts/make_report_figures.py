"""Figures of the English report (docs/report/REPORT.md) -> docs/report/figures/.

    python scripts/make_report_figures.py

Fig. 1 uses the corrected eps grid (0, 1], Fig. 3 the Choquet recursion for the non-monotone straddle,
Fig. 4/5 the unfiltered quotes with the (a)/(b) panels in the right order (thesis errata C2, C6, C7).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from dsbidask import R_DEFAULT, T_DEFAULT, S0_BID, b_up, bid_ask_trees
from dsbidask import plotting as P
from dsbidask.calibration import objective, model_quotes
from dsbidask.data import load_quotes
from dsbidask.pricing import european_tree, european_puts_bid_ask
from dsbidask.theory import spread_vs_epsilon

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "report" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "src" / "dsbidask" / "datasets"
DPI = 200
P.DPI = DPI
BID, ASK = "#2ca02c", "#d62728"


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=DPI); plt.close(fig)


# Fig. 1 -- spread vs epsilon on (0, 1]
eps = np.linspace(0.01, 1, 100)
for kind, tag in (("call", "a"), ("put", "b")):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for T in (5, 10, 15, 20):
        e, s = spread_vs_epsilon(kind, 100, 100, 1.05, 0.95, 1.01, T, eps)
        ax.plot(e, s, label=f"T = {T}")
    ax.set(xlabel=r"$\epsilon$", ylabel="Ask $-$ bid", title=f"European {kind}: theoretical spread"); ax.legend(); ax.grid(alpha=.3)
    save(fig, f"fig1{tag}_{kind}_spread.png")

# Fig. 2 -- stock tree
S, _, _ = bid_ask_trees(lambda s: s, S0=100, u=1.2, d=0.8, b_d=0.4 * 0.999, R=1.04, T=5)
P.plot_tree(S, r"Stock price tree, $S_0=100$, $u=1.2$, $d=0.8$", OUT / "fig2_stock_tree.png")

# Fig. 3 -- straddle by backward recursion
_, sb, sa = bid_ask_trees(lambda s: np.abs(s - 100), S0=100, u=1.2, d=0.8, b_d=0.4 * 0.999, R=1.04, T=5)
P.plot_tree(sb, "Straddle bid-price tree (backward recursion)", OUT / "fig3a_straddle_bid.png")
P.plot_tree(sa, "Straddle ask-price tree (backward recursion)", OUT / "fig3b_straddle_ask.png")

# Fig. 4 / 5 -- market quotes, unfiltered files
raw = {"call": pd.read_csv(DATA / "META_calls_2023-10-27.csv"), "put": pd.read_csv(DATA / "META_puts_2023-10-27.csv")}
for (kind, d), tag in zip(raw.items(), "ab"):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(d.strike, d.ask, s=14, c=ASK, label="Ask"); ax.scatter(d.strike, d.bid, s=14, c=BID, label="Bid")
    ax.set(xlabel="Strike $K$", ylabel="Price (USD)", title=f"META {kind}s ({len(d)} quotes)"); ax.legend()
    save(fig, f"fig4{tag}_{kind}_prices.png")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(d.strike, d.ask - d.bid, marker="+", c=ASK)
    ax.set(xlabel="Strike $K$", ylabel="Ask $-$ bid (USD)", title=f"META {kind}s: market spread"); 
    save(fig, f"fig5{tag}_{kind}_spread.png")

# Fig. 6 -- error surface, d = 1/u
q = load_quotes()
us, bds = np.linspace(1.022, 1.036, 80), np.linspace(0.47, 0.53, 80)
U, B = np.meshgrid(us, bds)
E = np.array([[objective(u, 1 / u, b, q) for u in us] for b in bds])
feas = (1 - np.array([b_up(u, 1 / u, R_DEFAULT) for u in us]))[None, :] >= B
u_opt, b_opt = 1.029428, 0.502787
fig = plt.figure(figsize=(5.5, 4.5)); ax = fig.add_subplot(projection="3d")
ax.plot_surface(U, B, np.minimum(E, 2250), cmap=cm.viridis, linewidth=0.2, edgecolor="k", rstride=4, cstride=4)
ax.set(xlabel="$u$", ylabel=r"$\widehat{b_d}$", zlabel=r"$E(u,\widehat{b_d})$"); ax.view_init(30, -115)
save(fig, "fig6a_error_surface.png")
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.contour(U, B, E, levels=np.geomspace(22, 400, 40), cmap="viridis")
ax.contourf(U, B, (~feas).astype(float), levels=[0.5, 1.5], colors=["orange"], alpha=0.4)
ax.plot(u_opt, b_opt, "o", color="tab:blue", label="PSO optimum"); ax.legend(loc="lower left")
ax.set(xlabel="$u$", ylabel=r"$\widehat{b_d}$", title=r"Contours of $E(u,\widehat{b_d})$ (orange: $\widehat{b_u}+\widehat{b_d}>1$)")
save(fig, "fig6b_contours.png")

# Fig. 7 -- calibrated put spread vs strike (free d, thesis parameters)
free = dict(u=1.02354592, d=0.96425885, b_d=0.39260853)
Ks = np.linspace(200, 400, 201)
pb, pa = european_puts_bid_ask(S0_BID, Ks, free["u"], free["d"], free["b_d"], R_DEFAULT, T_DEFAULT)
fig, ax = plt.subplots(figsize=(5.5, 3.8)); ax.plot(Ks, pa - pb, color="tab:blue")
ax.set(xlabel="Strike $K$", ylabel="Calibrated ask $-$ bid (USD)", title="European put: calibrated spread"); ax.grid(alpha=.3)
save(fig, "fig7_put_spread_vs_strike.png")

# Fig. 8 -- calibrated vs market quotes (d = 1/u fit, as in the text)
fx = dict(u=1.029428, b_d=0.502787); fx["d"] = 1 / fx["u"]
cb, ca, pb_, pa_ = model_quotes(fx["u"], fx["d"], fx["b_d"], q)
kc = q.K_call > 0
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, K, mod, mkt, title in ((axes[0, 0], q.K_call[kc], cb[kc], q.C_bid[kc], "Call bid"), (axes[0, 1], q.K_call[kc], ca[kc], q.C_ask[kc], "Call ask"),
                               (axes[1, 0], q.K_put, pb_, q.P_bid, "Put bid"), (axes[1, 1], q.K_put, pa_, q.P_ask, "Put ask")):
    o = np.argsort(K); ax.scatter(K, mkt, marker="+", color="tab:blue", label="Market"); ax.plot(K[o], mod[o], color="orange", lw=2, label="Model")
    ax.set(xlabel="Strike $K$", ylabel="Price (USD)", title=title); ax.legend()
save(fig, "fig8_calibration_fit.png")
print("saved", sorted(p.name for p in OUT.iterdir()))
