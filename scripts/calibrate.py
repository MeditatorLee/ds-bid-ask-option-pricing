"""Calibrate the DS bid-ask model to the bundled META quotes.

Examples
--------
python scripts/calibrate.py                         # one PSO run, free d, seed 12345 (thesis setting)
python scripts/calibrate.py --fixed-d               # d = 1/u
python scripts/calibrate.py --starts 50 --out results/free_d_multistart.csv
python scripts/calibrate.py --starts 50 --mode mid_spread --spread-weight 4
"""
import argparse
import json

from dsbidask.calibration import calibrate, calibrate_multistart, fit_report
from dsbidask.data import load_quotes

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--fixed-d", action="store_true", help="impose d = 1/u")
ap.add_argument("--maxiter", type=int, default=2000)
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--starts", type=int, default=1, help="number of PSO seeds (>1 enables multi-start + polish)")
ap.add_argument("--no-polish", action="store_true", help="skip the SLSQP polish in multi-start")
ap.add_argument("--wide-bounds", action="store_true", help="b_d in [0.01, 0.99] instead of the thesis [0.3, 0.6]")
ap.add_argument("--mode", default="price", choices=["price", "mid_spread", "inv_spread"])
ap.add_argument("--spread-weight", type=float, default=0.25)
ap.add_argument("--out", help="CSV file for the multi-start table")
a = ap.parse_args()

q = load_quotes()
kw = dict(free_d=not a.fixed_d, maxiter=a.maxiter, mode=a.mode, spread_weight=a.spread_weight)
if a.wide_bounds:
    kw["bd_bounds"] = (0.01, 0.99)
if a.starts > 1:
    best, table = calibrate_multistart(q, seeds=range(a.starts), do_polish=not a.no_polish, **kw)
    print(table.describe().loc[["min", "50%", "max"], ["error_pso", "error"]].round(3))
    if a.out:
        table.to_csv(a.out, index=False)
    res = {k: best[k] for k in ("u", "d", "b_d", "b_u", "error")}
else:
    res = calibrate(q, seed=a.seed, **kw)
rep = fit_report(res["u"], res["d"], res["b_d"], q)
print(json.dumps({**{k: float(v) for k, v in res.items()}, **rep}, indent=2))
