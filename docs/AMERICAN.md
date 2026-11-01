# American put extension: definition and what is verified

The thesis prices **European** claims. The American put in this repository is an **extension**, not a thesis result.

## Definition

On the recombining tree, with one-step lower/upper Choquet expectations `C_low`, `C_up` (see `dsbidask/choquet.py`):

```
bid:  Y_n    = max( phi(S_n), (1/R) * C_low[ Y_{n+1} | S_n ] ),   Y_T    = phi(S_T)
ask:  Ybar_n = max( phi(S_n), (1/R) * C_up [ Ybar_{n+1} | S_n ] ), Ybar_T = phi(S_T)
```

The ask uses the upper expectation because a writer must cover the worst case over the belief set;
this is a modelling choice and is **not** derived from the duality `-C[-.]` of thesis eq. (21) (which has no exercise decision).
No arbitrage or time-consistency theorem for this American rule is proved in the thesis.

## What is verified (`tests/test_american.py`)

| Property | Result |
|---|---|
| Equals the original authors' `AmericanPut.py` (bid, ask and number of early-exercise nodes), K in {250, 300, 350, 400}, T = 20 | exact (1e-10) |
| Put continuation values are non-increasing in the stock price at every node, so the authors' fixed weights `(1-b_d, b_d)` / `(b_u, 1-b_u)` coincide with the general Choquet recursion | holds |
| `bid <= ask`; American `>=` European; American `>=` intrinsic value; both non-decreasing in K | holds |
| Zero ambiguity (`b_u + b_d = 1`) equals an independent textbook CRR American put | exact |
| American call without dividends equals the European call (also under ambiguity) | holds |

## Not covered

* Non-monotone American payoffs: the recursion is well defined, but nothing is proved or tested about it.
* Whether bid/ask are the correct super/sub-hedging prices of an American claim: not established.
