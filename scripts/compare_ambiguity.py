"""Compare ambiguity parametrisations on the META quotes, in-sample and on held-out strikes.

Models: thesis (one-sided) vs symmetric, each with constant and state-dependent (Gaussian-in-log-moneyness) ambiguity.
Losses: ``price`` (thesis eq. 30) and ``mid_spread`` with a large spread weight (reproduce the market spreads).

Held-out protocol: 4 folds by strike position (calls and puts separately; the strike-0 stock quote is always in the
training set). Each fold is fitted on 3/4 of the strikes and evaluated on the remaining 1/4; metrics are pooled.

Run: ``python scripts/compare_ambiguity.py [--out results/ambiguity_comparison.csv] [--seeds 2] [--jobs 4]``
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from dsbidask import R_DEFAULT as R
from dsbidask.ambiguity import Ambiguity, calibrate_ambiguity, interval_report, model_quotes
from dsbidask.data import load_quotes

MODELS = {"thesis_const": ("thesis", False), "thesis_state": ("thesis", True),
          "sym_const": ("symmetric", False), "sym_state": ("symmetric", True)}
MODES = {"price": dict(mode="price"), "spread_w16": dict(mode="mid_spread", spread_weight=16.0)}
FOLDS = 4


def subset(q, call_idx, put_idx):
    return replace(q, K_call=q.K_call[call_idx], C_bid=q.C_bid[call_idx], C_ask=q.C_ask[call_idx],
                   K_put=q.K_put[put_idx], P_bid=q.P_bid[put_idx], P_ask=q.P_ask[put_idx])


def split(q, fold):
    """Train/test index sets; test = every FOLDS-th strike (offset ``fold``); strike 0 (stock) is always train."""
    ci, pi = np.arange(len(q.K_call)), np.arange(len(q.K_put))
    c_test = (ci > 0) & (ci % FOLDS == fold)
    p_test = pi % FOLDS == fold
    return (subset(q, ci[~c_test], pi[~p_test]), subset(q, ci[c_test], pi[p_test]))


def fit(job):
    name, mode_name, fold, seeds = job
    q = load_quotes()
    kind, sd = MODELS[name]
    train, test = (q, None) if fold < 0 else split(q, fold)
    r = calibrate_ambiguity(train, kind, sd, seeds=tuple(range(seeds)), **MODES[mode_name])
    amb = Ambiguity(r["a"], kind, r["width"] if sd else np.inf)
    out = dict(name=name, mode=mode_name, fold=fold, params=r)
    if test is not None:
        cb, ca, pb, pa = model_quotes(r["u"], r["d"], amb, test)
        out["held"] = (np.r_[cb, pb], np.r_[ca, pa], np.r_[test.C_bid, test.P_bid], np.r_[test.C_ask, test.P_ask])
    else:
        out["report"] = interval_report(r["u"], r["d"], amb, q)
    return out


def metrics(xb, xa, mb, ma):
    mid = (xb + xa) / 2
    return dict(rmse_price=float(np.sqrt(np.mean(np.r_[(xb - mb) ** 2, (xa - ma) ** 2]))),
                mean_model_spread=float(np.mean(xa - xb)), mean_market_spread=float(np.mean(ma - mb)),
                inside=float(np.mean((xb >= mb - 1e-9) & (xa <= ma + 1e-9))),
                overlap=float(np.mean((xb <= ma) & (xa >= mb))), mid_in=float(np.mean((mid >= mb) & (mid <= ma))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ambiguity_comparison.csv")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(n, m, f, a.seeds) for n, m in product(MODELS, MODES) for f in range(-1, FOLDS)]
    with Pool(a.jobs) as pool:
        res = pool.map(fit, jobs)
    rows = []
    for n, m in product(MODELS, MODES):
        sel = [r for r in res if r["name"] == n and r["mode"] == m]
        full = next(r for r in sel if r["fold"] < 0)
        held = [r["held"] for r in sel if r["fold"] >= 0]
        p = full["params"]
        base = dict(model=n, loss=m, u=p["u"], d=p["d"], a=p["a"], width=p["width"], b_u=p["b_u"], b_d=p["b_d"])
        rows.append(dict(**base, sample="in-sample", **full["report"]))
        rows.append(dict(**base, sample="held-out strikes", **metrics(*[np.concatenate(x) for x in zip(*held)])))
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(df.drop(columns=["u", "d", "b_u", "b_d"]).to_string(index=False))


if __name__ == "__main__":
    main()
