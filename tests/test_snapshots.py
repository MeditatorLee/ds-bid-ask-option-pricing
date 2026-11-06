from datetime import datetime, timezone

import pandas as pd
import pytest

from dsbidask.snapshots import collect, in_us_option_window, load_snapshots

# 2026-11-04 is a Wednesday; New York is on EST (UTC-5) -> 15:40 NY = 20:40 UTC
IN_WINDOW = datetime(2026, 11, 4, 20, 40, tzinfo=timezone.utc)
AFTER_CLOSE = datetime(2026, 11, 4, 22, 0, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 11, 7, 20, 40, tzinfo=timezone.utc)
# summer: 15:40 NY = 19:40 UTC (EDT)
IN_WINDOW_SUMMER = datetime(2026, 7, 15, 19, 40, tzinfo=timezone.utc)


def _chain(expiry):
    cols = ["contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"]
    calls = pd.DataFrame([[f"C{expiry}", 300.0, 5.0, 5.2, 5.1, 10, 100, 0.3]], columns=cols)
    puts = pd.DataFrame([[f"P{expiry}", 300.0, 4.0, 4.3, 4.1, 5, 50, 0.31]], columns=cols)
    return calls, puts, 300.1, 300.3


def _kw(tmp_path, **extra):
    return dict(outdir=tmp_path, list_expiries=lambda t: ["2026-11-06", "2026-12-18", "2027-06-18", "2026-11-04"],
                fetch_chain=lambda t, e: _chain(e), rate_fn=lambda: 0.045, **extra)


def test_window_logic():
    assert in_us_option_window(IN_WINDOW) and in_us_option_window(IN_WINDOW_SUMMER)
    assert not in_us_option_window(AFTER_CLOSE) and not in_us_option_window(SATURDAY)


def test_collect_refuses_outside_window_unless_forced(tmp_path):
    with pytest.raises(RuntimeError):
        collect("META", now=AFTER_CLOSE, **_kw(tmp_path))
    assert len(collect("META", now=AFTER_CLOSE, force=True, **_kw(tmp_path))) == 2


def test_collect_filters_by_dte_and_round_trips(tmp_path):
    files = collect("META", now=IN_WINDOW, min_dte=1, max_dte=60, **_kw(tmp_path))
    assert sorted(f.name for f in files) == ["2026-11-06.csv.gz", "2026-12-18.csv.gz"]  # same-day and +1y excluded
    df = load_snapshots(tmp_path, "meta")
    assert len(df) == 4 and set(df.side) == {"C", "P"}
    assert df.snapshot_date.iloc[0].isoformat() == "2026-11-04"
    assert (df.rate == 0.045).all() and (df.spot_bid == 300.1).all() and (df.spot_ask == 300.3).all()
    assert df["asof"].iloc[0] == "2026-11-04T20:40:00+00:00"


def test_collect_is_idempotent_and_overwrite_works(tmp_path):
    assert len(collect("META", now=IN_WINDOW, **_kw(tmp_path))) == 2
    assert collect("META", now=IN_WINDOW, **_kw(tmp_path)) == []
    assert len(collect("META", now=IN_WINDOW, overwrite=True, **_kw(tmp_path))) == 2


def test_load_snapshots_date_range_and_empty(tmp_path):
    assert load_snapshots(tmp_path, "META").empty
    collect("META", now=IN_WINDOW, **_kw(tmp_path))
    nxt = datetime(2026, 11, 5, 20, 40, tzinfo=timezone.utc)
    collect("META", now=nxt, **_kw(tmp_path))
    assert load_snapshots(tmp_path, "META", start="2026-11-05").snapshot_date.nunique() == 1
    assert load_snapshots(tmp_path, "META").snapshot_date.nunique() == 2
