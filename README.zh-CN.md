# DS 买卖价期权定价

[English](README.md) | **简体中文**

> 📄 **完整报告（英文）：[`docs/report/REPORT.md`](docs/report/REPORT.md)** —— 论文的英文版（理论、证明、META 校准），已纳入全部勘误。

基于 **Dempster-Shafer（DS）证据理论** 的动态买卖价期权定价模型：采用二叉树形式的 DS 乘性随机徘徊，
并用粒子群优化（PSO）对 META 期权报价进行校准（报价日 2023-09-29，到期日 2023-10-27，共 T = 20 个交易日）。

* 买价/卖价分别是 **Choquet 下期望/上期望**；`1 - b_u - b_d` 衡量模糊度，
  当 `b_u + b_d = 1` 时退化为经典 CRR 模型（买卖价差为零）。
* 对**单调合约函数**（看涨、看跌期权），价格有二项式求和的显式表达式（论文命题 4.1–4.2）。
* 对**任意合约函数**（跨式组合、美式提前行权），提供通用的逐步反向递推定价器。
* 实现并测试了复制投资组合及其自融资性质（命题 4.4）。

## 安装

```bash
pip install -e ".[dev]"            # 核心包 + pytest
pip install -e ".[fetch]"          # 可选：下载实时数据（yfinance、requests）
pip install -e ".[notebook]"       # 可选：jupyter，用于运行演示
pytest -q
```

## 快速开始

```python
import numpy as np
from dsbidask import *

u, d, b_d = 1.02354592, 0.96425885, 0.39260853        # 论文校准结果（d 自由）
print(european_call_bid_ask(300.11, 300, u, d, b_d, R_DEFAULT, 20))   # 显式公式
payoff = lambda s: np.abs(s - 300)                                     # 跨式：非单调
print(bid_ask(payoff, 300.11, u, d, b_d, R_DEFAULT, 20))               # 通用定价器
print(american_put_bid_ask(300.11, 350, u, d, b_d, R_DEFAULT, 20))     # 美式看跌（可提前行权）
```

完整演示见 [`notebooks/demo.ipynb`](notebooks/demo.ipynb)。校准：`python scripts/calibrate.py [--fixed-d]`；
重新生成图：`python scripts/make_figures.py`。
多起点：`python scripts/calibrate.py --starts 50 --out results/free_d.csv`（加 `--mode mid_spread --spread-weight 4` 可使用考虑价差的目标函数）。

## 目录结构与论文对应

| 路径 | 内容 | 论文位置 |
|---|---|---|
| `src/dsbidask/pricing.py` | 欧式买卖价显式公式（含对行权价向量化）、CRR、逐节点价格树 | 命题 4.1–4.2 |
| `src/dsbidask/choquet.py` | 通用反向递推定价（任意合约函数，可选美式行权）、提前行权节点 | 式 (18)–(21) |
| `src/dsbidask/replication.py` | 复制组合与自融资检验 | 4.2 节、命题 4.4 |
| `src/dsbidask/theory.py` | 买卖价差关于 ε 的函数；看跌价差关于行权价的函数（欧式与美式） | 图 1、图 7 |
| `src/dsbidask/calibration.py` | 向量化目标函数（论文式 (30) 及考虑价差的变体）、PSO、多起点 + SLSQP 精修 | 第 5 章、式 (30)–(31) |
| `src/dsbidask/data.py`、`datasets/` | CSV 读取与 4 个 META 期权数据文件 | 第 5 章 |
| `src/dsbidask/fetch.py`、`scripts/fetch_quotes.py` | **免费**实时数据下载（见下文；本仓库中未运行） | – |
| `src/dsbidask/plotting.py` | 全部绘图，仅用英文标签 | – |
| `docs/report/REPORT.md` | **英文报告**：论文的英文翻译（已含勘误），图在 `docs/report/figures/`，由 `scripts/make_report_figures.py` 生成 | – |
| `docs/thesis_corrections/` | 更正后的附录代码及中文论文的修改文本 | – |
| `docs/THESIS_ERRATA.md` | 论文勘误：数据文件、代码、表头/图题 | – |
| `docs/AMERICAN.md` | 美式看跌扩展的定义及已验证内容 | – |
| `results/` | 下文数字所依据的多起点表与目标函数对比（CSV） | – |
| `tests/` | 32 个测试（显式公式 vs 递推、CRR 极限、复制、美式 vs 原作者代码、校准） | – |
| `.github/workflows/tests.yml` | 在 Python 3.10/3.12/3.13 上运行 `pytest` 的 CI | – |

## 复现结果

| 校准方式 | u | d | b_d | b_u | E |
|---|---|---|---|---|---|
| `d = 1/u`（PSO，种子 12345） | 1.029428 | 0.971413 | 0.502787 | 0.496474 | 21.950091（与论文一致） |
| d 自由（论文结果点） | 1.023546 | 0.964259 | 0.392609 | 0.606494 | 15.4418（与论文一致） |

### 校准的稳健性（50 个 PSO 种子，各自用 SLSQP 精修；`results/multistart_*.csv`）

| 设定 | 最优 E | 中位 E | 最差 E | 说明 |
|---|---|---|---|---|
| `d = 1/u` | 21.9501 | 21.9501 | 21.9501 | 所有种子结果相同，问题性质良好 |
| d 自由，论文边界（`b_d` ∈ [0.3, 0.6]） | 11.69 | 17.06 | 4491（一次失败） | 最优点的 `b_d` **落在下界上**（u = 1.0192，d = 0.9561） |
| d 自由，宽边界（`b_d` ∈ [0.01, 0.99]） | 9.47 | 19.54 | 36.7 | 最优 `b_d` = 0.21，`b_u` = 0.789；50 个种子中仅 2 个到达 |

因此论文中的 15.44 只是 d 自由问题的一个典型局部最优，而非最小值；论文代码中的 `b_d ≥ 0.3` 边界是起作用的。
误差相近的解参数差别很大，应报告范围而不是单个种子的结果。

### 价格拟合 vs 价差拟合（`results/objective_comparison.csv`，每行 30 个种子）

价格拟合良好（每个报价 RMSE 约 0.28 美元），但模型价差（约 0.05 美元）明显窄于市场价差（约 0.20 美元）。
论文目标函数等于 `"mid_spread"` 目标函数在 `spread_weight = 0.25` 时的 2 倍。加大价差误差的权重可使模型价差升至约 0.12 美元，
代价是价格 RMSE 升至约 0.30 美元，但仍达不到市场水平：在此设定下单个 `b_d` 无法再现市场价差
（可能原因是同一个 `b_d` 要同时适用于所有行权价的看涨与看跌期权；此点未经检验）。
`"inv_spread"` 目标函数会使价差塌缩到接近零，不推荐。

## 实时数据（免费来源；提供脚本，**本仓库未运行**）

`scripts/fetch_quotes.py 代码 YYYY-MM-DD` 通过 **yfinance**（免费、无需密钥，论文所用来源）下载期权链，
按论文的流动性筛选（买卖价差 ≤ 1 美元），并从 **FRED**（序列 `DGS1MO`，免费 CSV，无需密钥）
打印 1 个月期美债收益率。`dsbidask.fetch.fetch_cboe_chain` 是 Cboe 延迟行情 JSON 的**未经测试**备选。
这些免费服务属于非官方渠道，可能有延迟、限流或格式变化；请在同一时刻的快照上校准。

## 局限

* 显式公式只适用于**单调合约函数**；其他情形请用 `dsbidask.choquet`。
* 美式扩展（行权价值与持有价值取较大者，并用下/上期望）**不是**论文结论；已验证的内容见 `docs/AMERICAN.md`。
* 仅一个报价日、一个到期日；`R` 为常数；无股息（META 2023 年不派息）。
* 校准的盒约束（`u <= 1.056`、`0.3 <= b_d <= 0.6`）来自论文代码，而非论文写出的问题 (31)。

## 此前建议中已完成的部分

稳健校准（多起点 + SLSQP 精修）、考虑价差的目标函数、已记录并测试的美式扩展、CI 工作流、类型标注与引用论文公式的文档字符串、
150 dpi 图片（打印用 `--dpi 300`）。

## 后续建议（仅记录，未实现）

1. **零买价报价**：拟合集中有 9 个看涨和 2 个看跌的 bid = 0，需决定剔除还是按删失数据处理。
2. **多到期日/多日期**：校准 `(u, d, b_d)` 的期限结构或随时间变化的 `b_d`；单个 `b_d` 无法再现市场价差。
3. **置信区间**：对报价做自助法得到参数不确定性（多起点的离散程度只是初步信号）。
4. **边界**：论证或取消起作用的 `b_d ≥ 0.3` 边界。
5. **连续时间**：将参数映射到 DS 几何布朗运动极限（论文展望）。
6. **发布**：`tests` 工作流已在 GitHub 上通过，准备好后可打标签发布（如 `v0.3.0`）。

## 致谢与许可证

以 **MIT 许可证**发布（见 [`LICENSE`](LICENSE)）：Copyright (c) 2024 Davide Petturiti（原始代码）、
Copyright (c) 2026 MeditatorLee（本衍生作品）。

本项目基于 [itsdavide/bid-ask-DS](https://github.com/itsdavide/bid-ask-DS)（MIT）的代码，该代码配套
A. Cinfrignini、D. Petturiti 与 B. Vantaggi 的论文 *Market consistent bid-ask option pricing under
Dempster-Shafer uncertainty*（Quantitative Finance，2024，doi:10.1080/14697688.2024.2353318）。
本仓库配套作者对该模型进行扩展的学位论文；`tests/test_american.py` 中的美式看跌递推是对原作者 `AmericanPut.py` 的移植，用作参照。

数据：仓库内的期权报价是 2023-09-29 的 Yahoo Finance 快照，仅用于复现论文结果；在此目的之外再分发市场数据前，请先查阅 Yahoo 的使用条款。
