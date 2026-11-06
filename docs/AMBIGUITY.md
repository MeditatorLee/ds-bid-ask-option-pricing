# Ambiguity extensions: symmetric and state-dependent (not in the thesis)

Code: `src/dsbidask/ambiguity.py`. Tests: `tests/test_ambiguity.py`. Experiment: `scripts/compare_ambiguity.py`
(`results/ambiguity_comparison.csv`) and `scripts/spread_by_moneyness.py` (`results/spread_by_moneyness.csv`).

## Models

With `q_u = (R-d)/(u-d)`, `q_d = 1-q_u` and total ambiguity `a = 1 - b_u - b_d`:

| kind | `b_u` | `b_d` | property |
|---|---|---|---|
| `thesis` | `q_u` | `q_d - a` | call bid and put ask are the CRR price; `C_bid - P_ask = S0 - K/R^T` exactly |
| `symmetric` | `q_u - a/2` | `q_d - a/2` | belief set centred on the CRR measure; at `T = 1` the synthetic-forward interval is centred (tested) |

`state-dependent`: `a(n, S) = a * exp(-0.5 (ln(S/S0)/width)^2)`; `width = inf` is the constant case.
Both kinds have the same number of parameters; `a = f * max_ambiguity` with `f in [0,1]` makes every point feasible.

## Verified (tests)

* `thesis` kind with constant `a` equals the closed forms of `pricing.py` (calls, puts) and the American put of `choquet.py`.
* `a = 0` gives CRR with zero spread, for both kinds.
* `bid <= ask`, prices non-negative; a state-dependent belief set is nested in the constant one, so its interval is narrower.
* Infeasible `a` raises.

## Result on the META quotes (2023-09-29 snapshot, 42 calls + 33 puts, free d, wide bounds)

Strike-wise 4-fold cross-validation (fit on 3/4 of strikes, pooled error on the held-out 1/4), `price` loss:

| model | in-sample RMSE | held-out RMSE | held-out mean model spread (market 0.202) |
|---|---|---|---|
| thesis, constant a | 0.249 | 0.278 | 0.065 |
| thesis, state-dependent | 0.246 | 0.275 | 0.061 |
| **symmetric, constant a** | **0.232** | **0.256** | **0.115** |
| symmetric, state-dependent | 0.232 | 0.256 | 0.114 |

* **Symmetric beats the thesis parametrisation**, in-sample and held-out (about 8 % lower RMSE held-out) and roughly
  doubles the model spread, with the same number of parameters.
* **State-dependent ambiguity does not help.** The fitted width sits on its upper bound (0.5, nearly constant) in three of
  four cases; the one interior solution (thesis kind, width 0.014) gains 0.003 RMSE in-sample. The hypothesis that
  an at-the-money bump is needed is **not supported**.
* Weighting the spread more (`mid_spread`, weight 16) barely changes the picture (see the CSV): `a` stays about 0.002.

## Where the missing spread is (symmetric, constant a)

| bucket (K / spot) | quotes | market spread | model spread |
|---|---|---|---|
| calls and puts, 0.95-1.05 | 11 | 0.49-0.51 | 0.42-0.44 |
| calls, > 1.20 | 26 | 0.082 | 0.005 |
| puts, < 0.80 | 18 | 0.127 | 0.009 |

Near the money the model reproduces the spread (about 85 %). In the wings it gives almost none, while the market
quotes 0.08-0.13 USD on options worth cents. Those wing spreads look like a **tick-size / fixed-cost floor**, not
ambiguity. This is an interpretation of one snapshot, not a tested claim.

## Caveats

* One snapshot, one expiry, 75 quotes; cross-validation by strike is not out-of-time validation.
* The mid error changes sign across moneyness (positive at the money, negative in the wings): a constant-`u, d`
  binomial cannot produce a smile. The extensions here do not address that.
* Differential evolution, 2 seeds per fit; optima of this landscape are known to be seed dependent (THESIS_ERRATA C5).
* No arbitrage or time-consistency result is proved for per-node masses.
