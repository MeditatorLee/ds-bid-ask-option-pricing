"""Rebuild notebooks/demo.ipynb (outputs stripped): python scripts/build_demo_notebook.py"""
import nbformat as nbf
from pathlib import Path

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md("# DS bid–ask option pricing: demo\n\nWalkthrough of the model from the thesis (Dempster–Shafer multiplicative random walk): 1) parameters, 2) European bid/ask and the CRR limit, 3) theoretical spreads, 4) bid/ask trees, 5) American put, 6) calibration to META options.")
code("import sys; sys.path.insert(0, '..')\nimport numpy as np, matplotlib.pyplot as plt\nfrom dsbidask import *\nfrom dsbidask.pricing import european_tree\nfrom dsbidask.theory import spread_vs_epsilon\nfrom dsbidask.data import load_quotes\nfrom dsbidask.calibration import sse, calibrate\nfrom dsbidask import plotting as P")
md("## 1. Parameters\n\nThe stock moves by factor `u` or `d` each period. `b_u = (R-d)/(u-d)` and `b_d` are the lower masses of the up and down moves; `1 - b_u - b_d >= 0` measures ambiguity. Zero ambiguity is the classical CRR model.")
code("S0, K, T = 100, 100, 5\nu, d, R = 1.2, 0.8, 1.04\nbu = b_up(u, d, R)\nb_d = 0.4 * 0.999\nprint(f'b_u = {bu:.4f}, b_d = {b_d:.4f}, ambiguity = {1 - bu - b_d:.4f}')")
md("## 2. European bid/ask prices\n\nFor monotone payoffs the bid/ask are binomial sums with different weights (`dsbidask/pricing.py`).")
code("print('call bid/ask:', european_call_bid_ask(S0, K, u, d, b_d, R, T))\nprint('put  bid/ask:', european_put_bid_ask(S0, K, u, d, b_d, R, T))\n\n# Zero ambiguity: both collapse to the CRR price\nprint('call at b_d = 1-b_u:', european_call_bid_ask(S0, K, u, d, 1 - bu, R, T))\nprint('CRR call           :', crr_price(lambda s: np.maximum(s - K, 0), S0, u, d, R, T))")
md("## 3. Theoretical spread vs ε  (b_d = ε(1 − b_u))")
code("curves = {f'T={t}': spread_vs_epsilon('call', 100, 100, 1.05, 0.95, 1.01, t) for t in (5, 10, 15, 20)}\nP.plot_spread_vs_epsilon(curves, 'Ask - bid', 'European call: theoretical spread');")
md("## 4. Bid and ask trees")
code("S, bid, ask = european_tree(lambda s: np.maximum(s - K, 0), S0, u, d, b_d, R, T, 'increasing')\nP.plot_tree(S, 'Stock tree'); P.plot_tree(bid, 'European call: bid tree'); P.plot_tree(ask, 'European call: ask tree');")
md("## 5. American put\n\nBackward induction with early exercise; American bid/ask dominate the European ones.")
code("ab, aa = american_put_bid_ask(S0, K, u, d, b_d, R, T)\neb, ea = european_put_bid_ask(S0, K, u, d, b_d, R, T)\nprint(f'American put bid/ask: {ab:.4f} / {aa:.4f}')\nprint(f'European put bid/ask: {eb:.4f} / {ea:.4f}')")
md("## 6. Calibration to META options (2023-09-29 → 2023-10-27)\n\nThe thesis parameters (free `d`, PSO seed 12345) are evaluated below. Re-running the optimiser takes several minutes: set `RERUN = True`.")
code("q = load_quotes()\nprint(len(q.K_call), 'calls,', len(q.K_put), 'puts')\nparams = dict(u=1.02354592, d=0.96425885, b_d=0.39260853)\nparams['b_u'] = b_up(params['u'], params['d'], R_DEFAULT)\nRERUN = False\nif RERUN:\n    params = calibrate(q, free_d=True)\nprint({k: round(float(v), 6) for k, v in params.items()})\nprint('SSE =', round(sse(params['u'], params['d'], params['b_d'], q), 4))")
code("P.plot_calibration_fit(q, params, R_DEFAULT, T_DEFAULT);")
md("## Limitations\n\n- The closed-form bid/ask require **monotone** payoffs; a straddle (call + put) is not covered.\n- The model-implied spread is narrower than the market spread: the thesis fits the shape of the curves, not their level.")
nb = nbf.v4.new_notebook(); nb.cells = c
nbf.write(nb, Path(__file__).resolve().parent.parent / "notebooks" / "demo.ipynb")
