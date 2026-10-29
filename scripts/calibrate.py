"""Run PSO calibration: python scripts/calibrate.py [--fixed-d] [--maxiter N]"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dsbidask.data import load_quotes
from dsbidask.calibration import calibrate

ap = argparse.ArgumentParser()
ap.add_argument("--fixed-d", action="store_true", help="impose d = 1/u")
ap.add_argument("--maxiter", type=int, default=2000)
a = ap.parse_args()
res = calibrate(load_quotes(), free_d=not a.fixed_d, maxiter=a.maxiter)
print(json.dumps({k: float(v) for k, v in res.items()}, indent=2))
