# DS 买卖价期权定价

[English](README.md) | **简体中文**

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

## 目录结构与论文对应

| 路径 | 内容 | 论文位置 |
|---|---|---|
| `src/dsbidask/pricing.py` | 欧式买卖价显式公式（含对行权价向量化）、CRR、逐节点价格树 | 命题 4.1–4.2 |
| `src/dsbidask/choquet.py` | 通用反向递推定价（任意合约函数，可选美式行权）、提前行权节点 | 式 (18)–(21) |
| `src/dsbidask/replication.py` | 复制组合与自融资检验 | 4.2 节、命题 4.4 |
| `src/dsbidask/theory.py` | 买卖价差关于 ε 的函数；看跌价差关于行权价的函数（欧式与美式） | 图 1、图 7 |
| `src/dsbidask/calibration.py` | 向量化的平方误差目标函数与 PSO（d 自由或 d = 1/u） | 第 5 章、式 (30)–(31) |
| `src/dsbidask/data.py`、`datasets/` | CSV 读取与 4 个 META 期权数据文件 | 第 5 章 |
| `src/dsbidask/fetch.py`、`scripts/fetch_quotes.py` | **免费**实时数据下载（见下文；本仓库中未运行） | – |
| `src/dsbidask/plotting.py` | 全部绘图，仅用英文标签 | – |
| `docs/THESIS_ERRATA.md` | 论文勘误：数据文件、代码、表头/图题 | – |
| `tests/` | 13 个测试（显式公式 vs 递推、CRR 极限、复制、论文数值） | – |

## 复现结果

| 校准方式 | u | d | b_d | b_u | E |
|---|---|---|---|---|---|
| `d = 1/u`（PSO，种子 12345） | 1.029428 | 0.971413 | 0.502787 | 0.496474 | 21.950091（与论文一致） |
| d 自由（论文结果点） | 1.023546 | 0.964259 | 0.392609 | 0.606494 | 15.4418（与论文一致） |

价格拟合良好（每个报价的 RMSE 约 0.32 美元），但模型价差（约 0.04 美元）明显窄于市场价差（约 0.2 美元）。
d 自由时的 PSO 结果**依赖随机种子**（已试的种子得到 E 在 10.3–19.5 之间，pyswarm 1.1.2），
所以 15.44 只是一个局部最优。

## 实时数据（免费来源；提供脚本，**本仓库未运行**）

`scripts/fetch_quotes.py 代码 YYYY-MM-DD` 通过 **yfinance**（免费、无需密钥，论文所用来源）下载期权链，
按论文的流动性筛选（买卖价差 ≤ 1 美元），并从 **FRED**（序列 `DGS1MO`，免费 CSV，无需密钥）
打印 1 个月期美债收益率。`dsbidask.fetch.fetch_cboe_chain` 是 Cboe 延迟行情 JSON 的**未经测试**备选。
这些免费服务属于非官方渠道，可能有延迟、限流或格式变化；请在同一时刻的快照上校准。

## 局限

* 显式公式只适用于**单调合约函数**；其他情形请用 `dsbidask.choquet`。
* 美式扩展（行权价值与持有价值取较大者，并用下/上期望）**不是**论文结论。
* 仅一个报价日、一个到期日；`R` 为常数；无股息（META 2023 年不派息）。
* 校准的盒约束（`u <= 1.056`、`0.3 <= b_d <= 0.6`）来自论文代码，而非论文写出的问题 (31)。

## 后续建议（仅记录，未实现）

1. **稳健校准**：多起点 PSO（多个种子，报告最优值与离散程度），再用局部优化器（`scipy.optimize.minimize`，约束 `b_u + b_d <= 1`）精修。
2. **考虑价差的目标函数**：按报价价差加权，或分别拟合中间价与价差，使模型能再现更宽的市场价差。
3. **零买价报价**：决定是剔除 bid = 0 的期权，还是按删失数据处理。
4. **多到期日/多日期**：校准 `(u, d, b_d)` 的期限结构或随时间变化的 `b_d`。
5. **置信区间**：对报价做自助法，得到参数不确定性。
6. **连续时间**：将参数映射到 DS 几何布朗运动极限（论文展望）。
7. **CI 与发布**：用 GitHub Actions 运行 `pytest`；确定许可证后打标签发布。

## 致谢与许可证

基于 Cinfrignini、Petturiti 与 Vantaggi 的论文 *Market consistent bid-ask option pricing under
Dempster-Shafer uncertainty*（Quantitative Finance，2024，doi:10.1080/14697688.2024.2353318）及其作者的原始代码；
本仓库配套作者对该模型进行扩展的学位论文。
**许可证：待项目完成后决定**（需先确认原始代码的许可证）。
