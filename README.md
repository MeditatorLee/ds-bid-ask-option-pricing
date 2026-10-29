# DS Bid-Ask Option Pricing

Dynamic bid–ask option pricing under **Dempster–Shafer (DS) uncertainty** with a binomial
DS multiplicative random walk, calibrated to META option quotes (2023-09-29 → 2023-10-27)
by Particle Swarm Optimisation. For monotone payoffs the Choquet bid/ask prices reduce to
closed-form binomial sums; when `b_u + b_d = 1` they collapse to the classical CRR price.

## Quick start
```bash
pip install -r requirements.txt
pytest -q
python scripts/calibrate.py            # free d  (u, d, b_d)
python scripts/calibrate.py --fixed-d  # d = 1/u
```
```python
from dsbidask import *
bid, ask = european_call_bid_ask(S0=300.11, K=300, u=1.0235, d=0.9643, b_d=0.3926, R=R_DEFAULT, T=20)
```

## Layout
- `dsbidask/` – `pricing.py` (European bid/ask, node-wise trees, American put, CRR), `theory.py` (spread vs ε), `data.py`, `calibration.py` (PSO), `plotting.py`, `config.py`
- `notebooks/demo.ipynb` – step-by-step demo (outputs stripped; rebuild with `scripts/build_demo_notebook.py`)
- `scripts/` – `calibrate.py`, `make_figures.py` (regenerates `figures/`)
- `data/` – Yahoo Finance option chains (`*_s_1_0_LOWER.csv` are the filtered sets used for calibration)
- `tests/` – pytest suite

## Reference result (free d, seed 12345)
u = 1.02354592, d = 0.96425885, b_d = 0.39260853, b_u ≈ 0.60649, SSE ≈ 15.44 (reproduced by the tests).

## Limitations
- The binomial formulas hold for **monotone** payoffs only (calls, puts); straddles etc. need the full Choquet integral.
- Calibration uses a single maturity and a single quote date.

## Credits
Based on Cinfrignini, Petturiti & Vantaggi, *Market consistent bid-ask option pricing under
Dempster–Shafer uncertainty*, Quantitative Finance (2024), doi:10.1080/14697688.2024.2353318, and the
authors' original code. This repository accompanies a thesis extending that model. **Add a LICENSE
once the original code's licence is confirmed.**
