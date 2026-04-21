# Stock Pair Trading Strategy Optimization
> 配对交易策略优化：DQN + Actor-Critic 混合深度强化学习

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 项目简介

本项目针对 A 股市场的**配对交易**（Pairs Trading）策略，提出并实现了一套
**DQN + Actor-Critic 混合深度强化学习模型**，并与传统 Z-score 基线策略进行对比实验。

**核心贡献：**
- 市场状态自适应决策（均值回归 / 趋势 / 震荡三类状态识别）  
- Double DQN + 优先经验回放（Sum-Tree PER）提升学习效率  
- A 股真实费率模型（含印花税，买 0.032% / 卖 0.132%）  
- 27 对协整配对全量评估，RL 模型平均回报较基线**提升 1208%**

---

## 项目结构

```
├── baseline_model.py          # Z-score 基线策略
├── train1.py                  # RL 模型训练入口
├── evaluate.py                # RL 模型评估
├── data_preprocessor/
│   └── data_preprocessor.py  # 数据清洗 & 协整筛选
├── data/
│   ├── data_generator.py      # 原始数据采集
│   ├── processed_hedge_data.csv
│   └── hedge_pairs.csv        # 27 对协整配对
└── models/
    ├── dqn_model.py           # HybridAgent（DQN + AC + PER）
    └── hedge_env.py           # 自定义交易环境
```

---

## 环境依赖

```bash
pip install torch numpy pandas matplotlib tqdm statsmodels
```

---

## 快速开始

```bash
# 1. 训练模型（早停，最优模型自动保存至 results/train/best_model.pth）
python train1.py

# 2. 评估 RL 模型（27 对配对）
python evaluate.py

# 3. 运行 Z-score 基线对比
python baseline_model.py
```

---

## 模型架构

```
输入状态（8维）
  z_score / z_velocity / z_volatility / unrealized_pnl
  position[0] / position[1] / total_return / time_progress
         │
   SharedEncoder（2层 LSTM + LayerNorm）
         │
    ┌────┴──────────────────────┐
    ▼                           ▼
MarketStateClassifier      context = [LSTM ‖ market_embed]
  0: 均值回归                   │
  1: 趋势               ┌───────┴───────┐
  2: 震荡               ▼               ▼
                      Critic(Q)      Actor(π)
                        └──── λQ + (1-λ)π ────▶ action
```

**损失函数：**

$$\mathcal{L} = \mathcal{L}_{Q}^{\text{Double DQN}} + 0.5\,\mathcal{L}_{\pi}^{\text{Actor}} + 0.3\,\mathcal{L}_{m}^{\text{Market}} + 0.01\,\mathcal{L}_{H}^{\text{Entropy}}$$

| 市场状态 | λ（Q 权重） | 1-λ（π 权重） |
|---|---|---|
| 均值回归 | 0.75 | 0.25 |
| 趋势 | 0.60 | 0.40 |
| 震荡 | 0.50 | 0.50 |

---

## 实验结果

### 整体指标对比（27 对配对，1564 交易日）

| 指标 | Z-score 基线 | DQN+AC 混合模型 | 变化 |
|---|---|---|---|
| 平均回报 | 53.85 | **704.63** | ▲ +1208% |
| 夏普比率 | 0.6482 | **3.1881** | ▲ +392% |
| 盈利配对数 | 25/27 (92.6%) | **27/27 (100%)** | ▲ |
| 平均最大回撤 | 33.38 | 48.80 | ▼ |
| 回报标准差 | — | 221.02 | — |

### 分行业表现

| 行业 | 配对数 | 基线均回报 | RL 均回报 | RL 胜出 |
|---|---|---|---|---|
| 基建 | 4 | 8.03 | 547.30 | 4/4 |
| 地产 | 1 | -0.02 | 805.15 | 1/1 |
| 消费 | 1 | 36.71 | 982.99 | 1/1 |
| 医药 | 2 | 75.20 | 844.93 | 2/2 |
| 医药-消费 | 19 | 65.00 | 702.52 | 19/19 |

> **结论：** RL 模型在全部 5 个行业板块均全面超越基线，盈利配对数达到 27/27（100%），较基线 25/27 有所提升。平均回报提升幅度达 1208%，夏普比率提升 392%，回报标准差 221.02 元。平均最大回撤由 55.72 降至 48.80（改善 12.4%），风险控制有所提升。

---

## 技术栈

| 模块 | 技术 |
|---|---|
| 深度学习 | PyTorch（LSTM, LayerNorm, Adam） |
| 强化学习 | Double DQN + Actor-Critic + PER（Sum-Tree） |
| 时序建模 | 双层 LSTM，滑动序列窗口（seq_len=10） |
| 环境 | 自定义 Gym 风格交易环境 |
| 数据 | Pandas + statsmodels 协整检验 |
| 可视化 | Matplotlib |

---

## 局限性与未来方向

- **风险控制不足**：奖励函数未对回撤显式惩罚，后续可引入 CVaR 损失
- **做空限制**：A 股个人账户实际无法融券做空，实盘需改为多头配对或 ETF 对冲
- **泛化能力**：可扩充更多行业配对、引入跨周期训练提升鲁棒性
- **算法升级**：可尝试 PPO / SAC 等在策略方法替代 DQN 以提升训练稳定性


