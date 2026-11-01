# Thesis errata (audit of the Chinese thesis against the data files and code)

Audit order, as requested: **1. data files → 2. code → 3. headers / captions**.
The thesis contains **no tables** (checked in the .docx), so "table headers" were audited as the CSV column
headers and the figure/table-like captions. Every number below was recomputed with this repository.

## 1. Data files

| # | Finding | Status |
|---|---|---|
| D1 | 67 calls / 42 puts originally; restricted set 42 calls + 33 puts (62.69 % / 78.57 %); spread ranges calls [0.05, 0.65], puts [0.02, 0.93]; stock quote 300.11 / 300.30; 20 trading days | **All consistent with the CSV files** |
| D2 | The restricted set contains **9 calls and 2 puts with bid = 0**. They are fitted like any other quote; the thesis does not mention it | Add one sentence, or drop zero-bid quotes and re-calibrate |
| D3 | "Theoretical stock ask 300.37" is correct for the d = 1/u fit (recomputed 300.367). For the free-d fit the implied ask is 300.43, wider than the market 300.30 | Optional remark |
| D4 | The four CSV files share identical 14-column headers; only cosmetic difference is `TRUE` vs `True` in `inTheMoney` | OK |

## 2. Code (thesis appendix)

| # | Finding | Fix |
|---|---|---|
| C1 | **Appendix "无限制PSO代码" is an exact copy of "部分限制PSO代码"** (2 parameters, `d = 1/u`). The free-d code (3 parameters) is missing, so the reported 15.441812 cannot be reproduced from the appendix | Replace with the 3-parameter code (`dsbidask.calibration.calibrate(free_d=True)`) |
| C2 | **Straddle example uses the monotone closed forms for a non-monotone payoff.** Appendix gives bid 35.342 > ask 35.309 (impossible). The thesis text itself says non-monotone payoffs need the backward recursion (19), (21). Correct values: **bid 35.302, ask 35.349** | Redraw Fig. 3 with `dsbidask.bid_ask_trees` (see `figures/tree_straddle_*.png`) |
| C3 | Same appendix block: docstring documents `plotPutTree`, code defines `plotStockTree`, example imports `plotYTree` (never defined); docstring says "European put on META, T = 20, K = 350" but the code runs a toy straddle (S0 = K = 100, T = 5) | Align docstring and code |
| C4 | Problem (31) omits the **box bounds used by the code**: `u ≤ 1.056`, `0.3 ≤ b_d ≤ 0.6`, `u ≥ R + 5e-5`. They matter: with `b_d ∈ [0.01, 0.99]` the optimiser lands elsewhere (E from 10.3 to 53 across seeds) | State the bounds in (31) |
| C5 | **Free-d calibration is seed dependent (50 seeds, PSO + SLSQP polish, pyswarm 1.1.2; `results/`).** d = 1/u is well behaved: all 50 seeds give u = 1.029428, b_d = 0.502787, E = 21.950091 (thesis value reproduced). For free d, 15.4418 is only a typical local optimum: with the thesis bounds the polished errors range 11.69 – 25.0 excluding one failed run (median 17.1; the best has b_d on the 0.3 bound, u = 1.0192, d = 0.9561); with `b_d ∈ [0.01, 0.99]` the best found is 9.47 (b_d = 0.21, b_u = 0.789) and only 2 of 50 seeds reach it. One thesis-bound run failed (E ≈ 4491) | Say "a local minimum found by PSO (seed 12345)"; report the multi-start range; consider relaxing the b_d bound |
| C6 | Spread-vs-ε code scans `np.arange(0, 1, 0.01)` = [0, 1) but Prop. 4.3 / text use ε ∈ (0, 1]; ε = 0 is outside the admissible set, ε = 1 (zero spread) is omitted | `np.linspace(0.01, 1, 100)` |
| C7 | Market plots read the **unfiltered** files (67 / 42 quotes) while calibration uses the restricted set; the thesis does not say which set Fig. 4–5 show | State the dataset |

## 3. Headers / captions / statements

| # | Location | Problem | Suggested text |
|---|---|---|---|
| H1 | Fig. 4 and Fig. 5 captions | "2023-9-29发行2023-10-23到期": maturity is **2023-10-27** (contract symbols `META231027…`); options are quoted, not "issued", on 9-29 | "2023-9-29报价、2023-10-27到期的META期权…" |
| H2 | Text after Fig. 6 | "与图1中的市场买卖价差表现一致": Fig. 1 is the theoretical spread vs ε; market spreads are Fig. 5 | "与图5(b)中的市场买卖价差表现一致" |
| H3 | Prop. 4.4 | Premise says "非增的合约函数" but the statement covers both cases | "设φ(x)是一个单调的合约函数" |
| H4 | Eqs. (22) and (25) | `\widehat{{b_u}^k}` puts the hat over the power | `(\widehat{b_u})^{k}` |
| H5 | Chapter 5, after (31) | "E … 是一个有最大值的非线性目标函数": the problem is a minimisation; presumably "非凸" / "有多个局部极小值" is meant | rephrase |
| H6 | Abstract / Conclusion | "模型有效拟合META期权买卖价差": prices fit well (RMSE 0.32 USD per quote) but model spreads average ≈ 0.04 USD vs ≈ 0.18–0.23 USD in the market | "拟合买卖价格，并再现价差随行权价变化的形状；价差水平偏窄" |
| H7 | Fig. 1 / Prop. 4.3 | OK: polynomial of degree T in ε; call spread ≫ put spread for small ε confirmed (e.g. T = 5: 18.96 vs 2.48) | – |

Items not edited in the .docx: formulas are Word equation objects, so the replacements above are given as text.
