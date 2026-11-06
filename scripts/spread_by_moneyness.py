"""Model vs market spread by moneyness bucket, for the fitted symmetric constant-ambiguity model.

Reads ``results/ambiguity_comparison.csv`` (made by ``scripts/compare_ambiguity.py``) and writes
``results/spread_by_moneyness.csv``. Strike-0 (stock) quote excluded; moneyness = K / spot mid.
"""
import pandas as pd

from dsbidask import R_DEFAULT as R
from dsbidask.ambiguity import Ambiguity, model_quotes
from dsbidask.data import load_quotes


def main(src="results/ambiguity_comparison.csv", out="results/spread_by_moneyness.csv"):
    r = pd.read_csv(src)
    r = r[(r.model == "sym_const") & (r["loss"] == "price") & (r["sample"] == "in-sample")].iloc[0]
    q = load_quotes()
    spot = (q.S0_bid + q.S0_ask) / 2
    cb, ca, pb, pa = model_quotes(r.u, r.d, Ambiguity(r.a, "symmetric"), q, R)
    rows = []
    for side, K, mb, ma, xb, xa in (("call", q.K_call, q.C_bid, q.C_ask, cb, ca), ("put", q.K_put, q.P_bid, q.P_ask, pb, pa)):
        for k, b, a, x, y in zip(K, mb, ma, xb, xa):
            if k > 0:
                rows.append((side, k / spot, a - b, y - x, (x + y) / 2 - (a + b) / 2))
    df = pd.DataFrame(rows, columns=["side", "moneyness", "market_spread", "model_spread", "mid_error"])
    df["bucket"] = pd.cut(df.moneyness, [0, 0.8, 0.95, 1.05, 1.2, 10], labels=["<0.80", "0.80-0.95", "0.95-1.05", "1.05-1.20", ">1.20"])
    g = df.groupby(["side", "bucket"], observed=True).agg(
        n=("moneyness", "size"), market_spread=("market_spread", "mean"), model_spread=("model_spread", "mean"),
        mean_mid_error=("mid_error", "mean")).round(4).reset_index()
    g.to_csv(out, index=False)
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
