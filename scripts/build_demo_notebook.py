"""Rebuild notebooks/demo.ipynb (outputs stripped): python scripts/build_demo_notebook.py"""
import nbformat as nbf
from pathlib import Path

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md("# DS bid–ask option pricing: demo\n\nWalkthrough of the model from the thesis (Dempster–Shafer multiplicative random walk): 1) parameters, 2) European bid/ask and the CRR limit, 3) theoretical spreads, 4) trees, 5) general pricer and straddle, 6) replication, 7) American put, 8) calibration, 9) robust calibration and spread-aware objectives.")
code("import numpy as np, matplotlib.pyplot as plt\nfrom dsbidask import *\nfrom dsbidask.pricing import european_tree\nfrom dsbidask.choquet import bid_ask, bid_ask_trees, early_exercise_nodes\nfrom dsbidask.replication import replicating_portfolio, is_self_financing\nfrom dsbidask.theory import put_spread_vs_strike\nfrom dsbidask.theory import spread_vs_epsilon\nfrom dsbidask.data import load_quotes\nfrom dsbidask.calibration import sse, calibrate, calibrate_multistart, fit_report\nfrom dsbidask import plotting as P")
md("## 1. Parameters\n\nThe stock moves by factor `u` or `d` each period. `b_u = (R-d)/(u-d)` and `b_d` are the lower masses of the up and down moves; `1 - b_u - b_d >= 0` measures ambiguity. Zero ambiguity is the classical CRR model.")
code("S0, K, T = 100, 100, 5\nu, d, R = 1.2, 0.8, 1.04\nbu = b_up(u, d, R)\nb_d = 0.4 * 0.999\nprint(f'b_u = {bu:.4f}, b_d = {b_d:.4f}, ambiguity = {1 - bu - b_d:.4f}')")
md("## 2. European bid/ask prices\n\nFor monotone payoffs the bid/ask are binomial sums with different weights (`dsbidask/pricing.py`).")
code("print('call bid/ask:', european_call_bid_ask(S0, K, u, d, b_d, R, T))\nprint('put  bid/ask:', european_put_bid_ask(S0, K, u, d, b_d, R, T))\n\n# Zero ambiguity: both collapse to the CRR price\nprint('call at b_d = 1-b_u:', european_call_bid_ask(S0, K, u, d, 1 - bu, R, T))\nprint('CRR call           :', crr_price(lambda s: np.maximum(s - K, 0), S0, u, d, R, T))")
md("## 3. Theoretical spread vs ε  (b_d = ε(1 − b_u))")
code("curves = {f'T={t}': spread_vs_epsilon('call', 100, 100, 1.05, 0.95, 1.01, t) for t in (5, 10, 15, 20)}\nP.plot_spread_vs_epsilon(curves, 'Ask - bid', 'European call: theoretical spread');")
md("## 4. Bid and ask trees")
code("S, bid, ask = european_tree(lambda s: np.maximum(s - K, 0), S0, u, d, b_d, R, T, 'increasing')\nP.plot_tree(S, 'Stock tree'); P.plot_tree(bid, 'European call: bid tree'); P.plot_tree(ask, 'European call: ask tree');")
md("## 5. General pricer: any payoff (thesis eqs. 19 and 21)\n\n`dsbidask.choquet` prices *any* payoff by backward recursion with one-step Choquet expectations. It agrees with the closed form for monotone payoffs, and it is the only valid method for a straddle.")
code("call = lambda s: np.maximum(s - K, 0)\nprint('recursion :', bid_ask(call, S0, u, d, b_d, R, T))\nprint('closed form:', european_call_bid_ask(S0, K, u, d, b_d, R, T))\n\nstraddle = lambda s: np.abs(s - K)\nprint('straddle (recursion):', bid_ask(straddle, S0, u, d, b_d, R, T))\nprint('straddle (monotone formula misapplied, thesis appendix):', european_bid_ask(straddle, S0, u, d, b_d, R, T, 'decreasing'), '<- bid > ask: invalid')")
code("S_, sb, sa = bid_ask_trees(straddle, S0, u, d, b_d, R, T)\nP.plot_tree(sb, 'Straddle bid tree'); P.plot_tree(sa, 'Straddle ask tree');")
md("## 6. Replication (thesis Section 4.2, Prop. 4.4)\n\nThe bid strategy of a non-decreasing payoff is self-financing; the ask strategy is not (and the reverse for non-increasing payoffs).")
code("S_, cb, ca = bid_ask_trees(call, S0, u, d, b_d, R, T)\nfor name, V in (('call bid', cb), ('call ask', ca)):\n    l0, l1, res = replicating_portfolio(S_, V, R)\n    print(name, 'self-financing:', is_self_financing(res), '| max |residual| = %.4f' % np.nanmax(np.abs(res)))")
md("## 7. American put\n\nBackward induction with early exercise (an extension: the thesis treats European claims). American bid/ask dominate the European ones; red nodes mark early exercise.")
code("ab, aa = american_put_bid_ask(S0, K, u, d, b_d, R, T)\neb, ea = european_put_bid_ask(S0, K, u, d, b_d, R, T)\nprint(f'American put bid/ask: {ab:.4f} / {aa:.4f}')\nprint(f'European put bid/ask: {eb:.4f} / {ea:.4f}')\nput = lambda s: np.maximum(K - s, 0)\nS_, Ab, Aa = bid_ask_trees(put, S0, u, d, b_d, R, T, american=True)\nfb, fa = early_exercise_nodes(put, S0, u, d, b_d, R, T)\nP.plot_tree(Ab, 'American put bid (red = exercise)', highlight=fb);")
md("## 8. Calibration to META options (2023-09-29 → 2023-10-27)\n\nThe thesis parameters (free `d`) are evaluated below. Re-running the optimiser takes about a second: set `RERUN = True`. Note that the free-`d` result depends on the random seed (see README).")
code("q = load_quotes()\nprint(len(q.K_call), 'calls (incl. stock as strike 0),', len(q.K_put), 'puts')\nparams = dict(u=1.02354592, d=0.96425885, b_d=0.39260853)\nparams['b_u'] = b_up(params['u'], params['d'], R_DEFAULT)\nRERUN = False\nif RERUN:\n    params = calibrate(q, free_d=True)\nprint({k: round(float(v), 6) for k, v in params.items()})\nprint('SSE =', round(sse(params['u'], params['d'], params['b_d'], q), 4))")
code("P.plot_calibration_fit(q, params, R_DEFAULT, T_DEFAULT);")
code("Ks = np.arange(200, 400.25, 0.25)\nsp = {lab: put_spread_vs_strike(S0_BID, Ks, params['u'], params['d'], params['b_d'], R_DEFAULT, T_DEFAULT, american=am)[2] for lab, am in (('European', False), ('American', True))}\nP.plot_put_spreads_vs_strike(Ks, sp);")
md("## 9. Robust calibration and spread-aware objectives\n\nOne PSO seed is not enough for free `d`: run many seeds and polish each. This takes a few seconds for 10 seeds; the README tables use 50.")
code("best, table = calibrate_multistart(q, free_d=True, seeds=range(10))\nprint(table[['seed', 'u', 'd', 'b_d', 'b_u', 'error_pso', 'error']].head(5).round(4).to_string(index=False))\nprint('error range over seeds: %.2f - %.2f (thesis point: 15.44)' % (table.error.min(), table.error.max()))")
code("for mode, w in (('price', 0.25), ('mid_spread', 16.0)):\n    b, _ = calibrate_multistart(q, seeds=range(10), mode=mode, spread_weight=w)\n    r = fit_report(b['u'], b['d'], b['b_d'], q)\n    print(f\"{mode:10s} price RMSE {r['rmse_price']:.3f}  model spread {r['mean_model_spread']:.3f}  market spread {r['mean_market_spread']:.3f}\")")
md("## Limitations\n\n- The closed-form bid/ask require **monotone** payoffs; a straddle (call + put) is not covered.\n- Prices fit well (RMSE about 0.32 USD per quote) but model spreads (about 0.04 USD) are much narrower than market spreads (about 0.2 USD).\n- The American extension is not part of the thesis theory.")
nb = nbf.v4.new_notebook(); nb.cells = c
nbf.write(nb, Path(__file__).resolve().parent.parent / "notebooks" / "demo.ipynb")
