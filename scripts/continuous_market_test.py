"""Test the continuous-time DS formulas (``dsbidask.continuous``) on the META quotes.

Parts: [A] fit + strike-wise 4-fold CV, [B] underlying spread implied by the options, [C] option spread implied by the
underlying, [D] the structural bound "option spread <= underlying spread", [E] what the market spread tracks.
Writes ``results/continuous_market_test.csv`` (part A). Takes about 1 minute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm

from compare_ambiguity import split
from dsbidask.continuous import R_CONT as r, T_YEARS as T, _loss, calibrate_continuous, model_quotes, shift, stock_quote
from dsbidask.data import load_quotes

q = load_quotes()
S_mid = (q.S0_bid + q.S0_ask) / 2
stock_spread = q.S0_ask - q.S0_bid


def arrays(qs, sigma, theta, kind, floor=0.0):
    cb, ca, pb, pa = model_quotes(qs, sigma, theta, kind, floor=floor)
    k = qs.K_call > 0
    return np.r_[cb[k], pb], np.r_[ca[k], pa], np.r_[qs.C_bid[k], qs.P_bid], np.r_[qs.C_ask[k], qs.P_ask]


def metrics(xb, xa, mb, ma):
    mid = (xb + xa) / 2
    return dict(rmse=float(np.sqrt(np.mean(np.r_[(xb - mb) ** 2, (xa - ma) ** 2]))), spread=float(np.mean(xa - xb)),
                overlap=float(np.mean((xb <= ma) & (xa >= mb))), mid_in=float(np.mean((mid >= mb) & (mid <= ma))))


def fit_bs(qs):
    """Black-Scholes (theta = 0), flat sigma."""
    f = lambda z: _loss([z[0], 0.0], "symmetric", qs, r, T, False, False)
    return float(differential_evolution(f, [(0.05, 1.5)], seed=0, tol=1e-12, polish=True).x[0])


def part_a(out):
    models = [("BS (theta=0)", "symmetric", False, True), ("cont thesis", "thesis", False, False),
              ("cont symmetric", "symmetric", False, False), ("cont symmetric + floor", "symmetric", True, False)]
    rows = []
    for name, kind, fl, bs in models:
        def run(qs, seeds):
            if bs:
                return fit_bs(qs), 0.0, 0.0
            c = calibrate_continuous(qs, kind, with_floor=fl, seeds=seeds)
            return c["sigma"], c["theta"], c.get("floor", 0.0)
        s, th, f0 = run(q, (0, 1, 2))
        ins = metrics(*arrays(q, s, th, kind, f0))
        held = [[], [], [], []]
        for fold in range(4):
            tr, te = split(q, fold)
            sf, tf, ff = run(tr, (0, 1))
            for i, a in enumerate(arrays(te, sf, tf, kind, ff)):
                held[i].append(a)
        cv = metrics(*[np.concatenate(h) for h in held])
        rows.append(dict(model=name, sigma=s, theta=th, floor=f0, in_rmse=ins["rmse"], in_spread=ins["spread"],
                         in_overlap=ins["overlap"], cv_rmse=cv["rmse"], cv_spread=cv["spread"], cv_overlap=cv["overlap"],
                         cv_mid_in=cv["mid_in"]))
    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"stock quote {q.S0_bid}/{q.S0_ask} (spread {stock_spread:.2f}), T = {T}, r = {r:.4f}")
    print("[A] fit and strike-wise 4-fold CV (price loss includes the stock quote; metrics exclude it)")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    mk = np.mean(np.r_[(q.C_ask - q.C_bid)[q.K_call > 0], q.P_ask - q.P_bid])
    print(f"market mean spread = {mk:.3f}")
    return df


def part_bcd():
    print("\n[B] underlying spread implied by the OPTIONS (stock quote dropped from the fit)")
    for kind in ("thesis", "symmetric"):
        c = calibrate_continuous(q, kind, drop_stock=True)
        sb, sa = stock_quote(q.S0_bid, c["theta"], T, kind)
        print(f"  {kind:9s}: sigma={c['sigma']:.3f} theta={c['theta']:.4f}/yr -> model stock {sb:.2f}/{sa:.2f} "
              f"(spread {sa - sb:.2f}) vs observed {q.S0_bid:.2f}/{q.S0_ask:.2f} (spread {stock_spread:.2f})")

    print("\n[C] option spread implied by the UNDERLYING (theta from the stock spread, sigma from option mids)")
    for kind in ("thesis", "symmetric"):
        mb, ma = shift(kind, "call")
        th = np.log(q.S0_ask / q.S0_bid) / ((ma - mb) * T)
        # sigma fitted on the option MIDS only (stock quote excluded)
        def mid_loss(z):
            cb, ca, pb, pa = model_quotes(q, z[0], th, kind)
            k = q.K_call > 0
            return float(np.sum(((cb + ca) / 2 - (q.C_bid + q.C_ask) / 2)[k] ** 2) + np.sum(((pb + pa) / 2 - (q.P_bid + q.P_ask) / 2) ** 2))
        s = float(differential_evolution(mid_loss, [(0.05, 1.5)], seed=0, tol=1e-12, polish=True).x[0])
        cb, ca, pb, pa = model_quotes(q, s, th, kind)
        atm_c = (np.abs(q.K_call / S_mid - 1) < 0.05) & (q.K_call > 0); atm_p = np.abs(q.K_put / S_mid - 1) < 0.05
        mk = np.r_[(q.C_ask - q.C_bid)[atm_c], (q.P_ask - q.P_bid)[atm_p]]; md = np.r_[(ca - cb)[atm_c], (pa - pb)[atm_p]]
        print(f"  {kind:9s}: theta_stock={th:.4f}/yr sigma={s:.3f}: mean ATM spread model {md.mean():.3f} vs market {mk.mean():.3f} (n={len(mk)})")

    print("\n[D] bound: option spread <= underlying spread (|delta| <= 1)")
    cs = (q.C_ask - q.C_bid)[q.K_call > 0]; ps = q.P_ask - q.P_bid
    ok_c = q.C_bid[q.K_call > 0] > 0; ok_p = q.P_bid > 0
    sp = np.r_[cs[ok_c], ps[ok_p]]
    print(f"  stock spread {stock_spread:.2f}; quotes with bid>0: {len(sp)}; spread > stock spread: {np.mean(sp > stock_spread):.0%} "
          f"({int(np.sum(sp > stock_spread))}/{len(sp)}); median {np.median(sp):.2f}, max {sp.max():.2f}")
    return sp, ok_c, ok_p


def part_e(sp, ok_c, ok_p):
    print("\n[E] what the market spread tracks (bid>0), sigma from the symmetric continuous fit")
    s = calibrate_continuous(q, "symmetric")["sigma"]
    def greeks(K, call):
        d1 = (np.log(S_mid / K) + (r + s * s / 2) * T) / (s * np.sqrt(T))
        return np.abs(norm.cdf(d1) - (0 if call else 1)), S_mid * norm.pdf(d1) * np.sqrt(T)
    dc, vc = greeks(q.K_call[q.K_call > 0][ok_c], True); dp, vp = greeks(q.K_put[ok_p], False)
    D, V, one = np.r_[dc, dp], np.r_[vc, vp], np.ones(len(sp))
    for lab, X in (("|delta| only (model)", [D]), ("|delta| + const", [D, one]), ("vega only", [V]), ("vega + const", [V, one]),
                   ("|delta| + vega + const", [D, V, one])):
        X = np.column_stack(X); b, *_ = np.linalg.lstsq(X, sp, rcond=None)
        print(f"  {lab:24s} R2={1 - (sp - X @ b).var() / sp.var():5.2f} coef={np.round(b, 4)}")
    print(f"  (the model implies slope on |delta| = underlying spread {stock_spread:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/continuous_market_test.csv")
    a = ap.parse_args()
    part_a(a.out)
    sp, ok_c, ok_p = part_bcd()
    part_e(sp, ok_c, ok_p)
