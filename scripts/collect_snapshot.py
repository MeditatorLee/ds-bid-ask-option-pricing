"""Store today's option chains (free Yahoo data + FRED 1-month rate) under data/snapshots/.

Not run in this repository's tests (needs network). Typical use, once per US trading day near the close::

    pip install -e ".[fetch]"
    python scripts/collect_snapshot.py META --max-dte 60

Outside 15:30-15:59 New York time on weekdays the script refuses to run unless ``--force`` is given.
"""
import argparse

from dsbidask.snapshots import DEFAULT_ROOT, collect


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--outdir", default=str(DEFAULT_ROOT))
    ap.add_argument("--min-dte", type=int, default=1)
    ap.add_argument("--max-dte", type=int, default=60)
    ap.add_argument("--force", action="store_true", help="ignore the market-hours window")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for t in a.tickers:
        files = collect(t, a.outdir, min_dte=a.min_dte, max_dte=a.max_dte, force=a.force, overwrite=a.overwrite)
        print(f"{t}: wrote {len(files)} file(s)")


if __name__ == "__main__":
    main()
