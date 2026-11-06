"""Fetch live option chains and the risk-free rate from FREE sources.

NOTE: nothing in this module is executed by the test suite or the demo notebook; it needs
network access and the third-party services may change or rate-limit at any time.

Sources
-------
* Yahoo Finance via ``yfinance`` (free, no key; what the thesis used). Install: ``pip install ".[fetch]"``.
* FRED (St. Louis Fed) CSV export for the 1-month Treasury yield ``DGS1MO`` (free, no key).
* Cboe delayed quotes JSON (free, delayed ~15 min): UNTESTED fallback, the endpoint format
  may differ from what is assumed below.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

KEEP = ["contractSymbol", "lastTradeDate", "strike", "lastPrice", "bid", "ask", "change",
        "percentChange", "volume", "openInterest", "impliedVolatility", "inTheMoney",
        "contractSize", "currency"]


def list_expiries(ticker: str) -> Tuple[str, ...]:
    """All listed expiry dates (YYYY-MM-DD) of a ticker from Yahoo Finance."""
    import yfinance as yf
    return tuple(yf.Ticker(ticker).options)


def fetch_yahoo_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float], Optional[float]]:
    """Return ``(calls, puts, spot_bid, spot_ask)`` for one expiry (YYYY-MM-DD) from Yahoo Finance."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    chain = t.option_chain(expiry)
    info = t.info or {}
    return (chain.calls[KEEP], chain.puts[KEEP], info.get("bid"), info.get("ask"))


def fetch_cboe_chain(ticker: str) -> pd.DataFrame:
    """UNTESTED fallback: Cboe delayed-quotes JSON. Returns a DataFrame with an OCC ``option`` column."""
    import requests
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker.upper()}.json"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return pd.DataFrame(r.json()["data"]["options"])


def fetch_risk_free_rate(series: str = "DGS1MO") -> float:
    """Latest annualised yield (as a decimal, e.g. 0.0555) of a FRED Treasury series (no API key)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    df = pd.read_csv(url)
    df = df[pd.to_numeric(df.iloc[:, 1], errors="coerce").notna()]
    return float(df.iloc[-1, 1]) / 100.0


def liquid_subset(df: pd.DataFrame, max_spread: float = 1.0) -> pd.DataFrame:
    """Thesis filter: keep options whose bid-ask spread is at most ``max_spread`` dollars."""
    return df[(df["ask"] - df["bid"]) <= max_spread].copy()


def save_snapshot(calls: pd.DataFrame, puts: pd.DataFrame, outdir: Union[str, Path], ticker: str, expiry: str) -> None:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    calls.to_csv(outdir / f"{ticker}_calls_{expiry}.csv", index=False)
    puts.to_csv(outdir / f"{ticker}_puts_{expiry}.csv", index=False)
