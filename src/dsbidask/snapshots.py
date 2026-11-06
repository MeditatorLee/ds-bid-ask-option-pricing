"""Daily option-chain snapshot store: the data layer for a backtest (EXTENSION, not in the thesis).

Free sources only give the *current* chain, so a history has to be collected forward in time.
:func:`collect` writes one gzipped CSV per ``(date, ticker, expiry)`` and :func:`load_snapshots` reads them back.

Layout::

    data/snapshots/<TICKER>/<YYYY-MM-DD>/<expiry>.csv.gz     # calls and puts stacked, column ``side`` in {C, P}

Every row carries the snapshot time (``asof``, UTC), the underlying bid/ask (``spot_bid``/``spot_ask``) and the
risk-free rate (``rate``, annual decimal) observed at that time, so that every later calibration is point-in-time.

Network access happens only through the injected callables (defaults in :mod:`dsbidask.fetch`), which makes the
logic testable offline (``tests/test_snapshots.py``).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union
from zoneinfo import ZoneInfo

import pandas as pd

from . import fetch

PathLike = Union[str, Path]
NY = ZoneInfo("America/New_York")
DEFAULT_ROOT = Path("data/snapshots")


def in_us_option_window(now: Optional[datetime] = None, start: str = "15:30", end: str = "15:59") -> bool:
    """True on a weekday between ``start`` and ``end`` New York time (default: the last half hour before the close).

    Quotes taken after the close are often stale or zero (Yahoo), so the collector refuses to run outside the window
    unless forced. US market holidays are *not* detected: a snapshot on a holiday should be dropped in cleaning.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(NY)
    if now.weekday() >= 5:
        return False
    hm = now.strftime("%H:%M")
    return start <= hm <= end


def collect(ticker: str, outdir: PathLike = DEFAULT_ROOT, *, now: Optional[datetime] = None, min_dte: int = 1,
            max_dte: int = 60, force: bool = False, overwrite: bool = False,
            list_expiries: Callable[[str], Sequence[str]] = fetch.list_expiries,
            fetch_chain: Callable = fetch.fetch_yahoo_chain,
            rate_fn: Callable[[], float] = fetch.fetch_risk_free_rate) -> List[Path]:
    """Fetch every expiry with ``min_dte <= days to expiry <= max_dte`` and store it. Returns the files written.

    Skips a ``(day, expiry)`` already stored unless ``overwrite``. Raises if called outside
    :func:`in_us_option_window` without ``force``.
    """
    now = now or datetime.now(timezone.utc)
    if not force and not in_us_option_window(now):
        raise RuntimeError("outside the US option snapshot window (15:30-15:59 New York, weekdays); use force=True")
    day = now.astimezone(NY).date()
    rate = float(rate_fn())
    folder = Path(outdir) / ticker.upper() / day.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for expiry in list_expiries(ticker):
        dte = (date.fromisoformat(expiry) - day).days
        if not (min_dte <= dte <= max_dte):
            continue
        path = folder / f"{expiry}.csv.gz"
        if path.exists() and not overwrite:
            continue
        calls, puts, s_bid, s_ask = fetch_chain(ticker, expiry)
        df = pd.concat([calls.assign(side="C"), puts.assign(side="P")], ignore_index=True)
        df.insert(0, "asof", now.astimezone(timezone.utc).isoformat(timespec="seconds"))
        df.insert(1, "expiry", expiry)
        df["spot_bid"], df["spot_ask"], df["rate"] = s_bid, s_ask, rate
        df.to_csv(path, index=False, compression="gzip")
        written.append(path)
    return written


def load_snapshots(root: PathLike = DEFAULT_ROOT, ticker: str = "", start: Optional[str] = None,
                   end: Optional[str] = None) -> pd.DataFrame:
    """Read stored snapshots of one ticker into one frame with a ``snapshot_date`` column (date), sorted.

    ``start``/``end`` are inclusive ISO dates. Returns an empty frame if nothing is stored.
    """
    base = Path(root) / ticker.upper()
    frames = []
    for folder in sorted(p for p in base.glob("*") if p.is_dir()):
        if (start and folder.name < start) or (end and folder.name > end):
            continue
        for f in sorted(folder.glob("*.csv.gz")):
            frames.append(pd.read_csv(f).assign(snapshot_date=folder.name))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df.sort_values(["snapshot_date", "expiry", "side", "strike"]).reset_index(drop=True)
