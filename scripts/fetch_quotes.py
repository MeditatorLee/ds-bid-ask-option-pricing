"""Download a fresh option snapshot (NOT run in this repo's tests/CI).

Example:
    pip install ".[fetch]"
    python scripts/fetch_quotes.py META 2026-12-18 --out my_data
Then calibrate on it with ``dsbidask.data.load_quotes(calls=..., puts=..., spot=(bid, ask))``.
"""
import argparse
from dsbidask.fetch import fetch_yahoo_chain, fetch_risk_free_rate, liquid_subset, save_snapshot

ap = argparse.ArgumentParser()
ap.add_argument("ticker"); ap.add_argument("expiry", help="YYYY-MM-DD")
ap.add_argument("--out", default="live_data"); ap.add_argument("--max-spread", type=float, default=1.0)
a = ap.parse_args()
calls, puts, s_bid, s_ask = fetch_yahoo_chain(a.ticker, a.expiry)
save_snapshot(calls, puts, a.out, a.ticker, a.expiry)
save_snapshot(liquid_subset(calls, a.max_spread), liquid_subset(puts, a.max_spread), a.out, a.ticker, a.expiry + "_s_1_0_LOWER")
print(f"saved to {a.out}/; spot bid/ask = {s_bid}/{s_ask}; 1M T-bill yield = {fetch_risk_free_rate():.4%}")
