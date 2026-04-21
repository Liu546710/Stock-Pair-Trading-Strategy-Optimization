"""
基于Z-score的传统配对交易策略基准模型
用于与强化学习模型进行性能比较
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib as mpl

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

plt.rcParams['font.size'] = 12  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 14  # 设置标题字体大小
plt.rcParams['figure.titlesize'] = 16  # 设置图表标题字体大小
plt.rcParams['figure.figsize'] = [15, 10]  # 设置默认图表大小

class ZScoreStrategy:
    def __init__(self, entry_threshold=1.0, exit_threshold=0.0, 
                 stop_loss=2.5,
                 buy_cost_rate=0.00032,    # 买入：经纪 0.03% + 过户费 0.002%
                 sell_cost_rate=0.00132):  # 卖出：经纪 0.03% + 过户费 0.002% + 印花税 0.1%
        """
        初始化Z-score策略

        参数:
        entry_threshold: 开仓阈値，当|z-score| > entry_threshold时开仓
        exit_threshold:  平仓阈値，当|z-score| < exit_threshold时平仓
        stop_loss:       止损阈値，当|z-score| > stop_loss时止损
        buy_cost_rate:   买入单边费率（经纪+过户费）
        sell_cost_rate:  卖出单边费率（经纪+过费+印花税）
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.buy_cost_rate = buy_cost_rate
        self.sell_cost_rate = sell_cost_rate
        
        # 仓位状态
        self.position_1 = 0
        self.position_2 = 0
        
        # 交易记录
        self.trades = []
        self.positions_history = []
        self.rewards_history = []
        self.prices_history = []
        self.actions_history = []
        
    def reset(self):
        """重置策略状态"""
        self.position_1 = 0
        self.position_2 = 0
        self.trades = []
        self.positions_history = []
        self.rewards_history = []
        self.prices_history = []
        self.actions_history = []
        
    def decide_action(self, z_score, price_ratio):
        """
        根据z-score决定交易动作
        返回值：0=持仓, 1=买入资产1/卖出资产2, 2=卖出资产1/买入资产2
        """
        # 当前没有仓位
        if self.position_1 == 0 and self.position_2 == 0:
            if z_score > self.entry_threshold:
                # z偏高 → 资产1相对高估 → 卖出资产1/买入资产2
                return 2
            elif z_score < -self.entry_threshold:
                # z偏低 → 资产1相对低估 → 买入资产1/卖出资产2
                return 1

        # 已有多头资产1/空头资产2（因z < -entry开仓）
        elif self.position_1 > 0 and self.position_2 < 0:
            if z_score >= -self.exit_threshold:  # z回归到0附近，平仓
                return 2
            elif z_score < -self.stop_loss:      # z继续下探超过止损，止损
                return 2

        # 已有空头资产1/多头资产2（因z > +entry开仓）
        elif self.position_1 < 0 and self.position_2 > 0:
            if z_score <= self.exit_threshold:   # z回归到0附近，平仓
                return 1
            elif z_score > self.stop_loss:       # z继续上冲超过止损，止损
                return 1

        return 0
    
    def execute_action(self, action, price_1, price_2, z_score):
        """
        执行交易动作并计算回报
        """
        old_position_1 = self.position_1
        old_position_2 = self.position_2
        reward = 0
        
        # 记录价格和动作
        self.prices_history.append((price_1, price_2))
        self.actions_history.append(action)
        
        # 买入资产1，卖出资产2
        if action == 1:
            # 先平已有相反方向仓位（空1/多2）
            if self.position_1 < 0 and self.trades:
                # 平空1：买入平仓，扣买入费率
                close_return = -self.position_1 * (self.trades[-1][0] - price_1)
                close_return -= self.buy_cost_rate * price_1
                reward += close_return
                self.position_1 = 0
            if self.position_2 > 0 and self.trades:
                # 平多2：卖出平仓，扣卖出费率
                close_return = self.position_2 * (price_2 - self.trades[-1][1])
                close_return -= self.sell_cost_rate * price_2
                reward += close_return
                self.position_2 = 0
            # 开新仓（买asset1 + 卖asset2）
            if self.position_1 == 0 and self.position_2 == 0:
                self.position_1 = 1
                self.position_2 = -1
                self.trades.append((price_1, price_2))
                reward -= (self.buy_cost_rate * price_1 + self.sell_cost_rate * price_2)

        # 卖出资产1，买入资产2
        elif action == 2:
            # 先平已有相反方向仓位（多1/空2）
            if self.position_1 > 0 and self.trades:
                # 平多1：卖出平仓，扣卖出费率
                close_return = self.position_1 * (price_1 - self.trades[-1][0])
                close_return -= self.sell_cost_rate * price_1
                reward += close_return
                self.position_1 = 0
            if self.position_2 < 0 and self.trades:
                # 平空2：买入平仓，扣买入费率
                close_return = -self.position_2 * (self.trades[-1][1] - price_2)
                close_return -= self.buy_cost_rate * price_2
                reward += close_return
                self.position_2 = 0
            # 开新仓（卖asset1 + 买asset2）
            if self.position_1 == 0 and self.position_2 == 0:
                self.position_1 = -1
                self.position_2 = 1
                self.trades.append((price_1, price_2))
                reward -= (self.sell_cost_rate * price_1 + self.buy_cost_rate * price_2)

        # 记录仓位和回报
        self.positions_history.append((self.position_1, self.position_2))
        self.rewards_history.append(reward)
        return reward

def plot_pair_performance(dates, prices_1, prices_2, z_scores, strategy, 
                         cumulative_returns, pair_idx, stock1, stock2, 
                         total_reward, sharpe_ratio, results_dir,
                         entry_threshold=1.0, exit_threshold=0.0, stop_loss=2.5):
    """绘制配对交易的性能图表"""
    plt.figure(figsize=(15, 12))
    
    # 转换日期格式并选择每三个月显示一次
    dates_pd = pd.to_datetime(dates)
    # 每三个月选择一个点
    date_indices = np.arange(0, len(dates), 60)  # 假设每月约20个交易日，60天约为3个月
    
    # 设置x轴刻度
    def format_dates(ax):
        ax.set_xticks(date_indices)
        ax.set_xticklabels([dates_pd[i].strftime('%Y-%m') for i in date_indices], 
                          rotation=45, ha='right')
    
    # 价格图
    ax1 = plt.subplot(4, 1, 1)
    plt.plot(range(len(dates)), prices_1, label=f'资产1 ({stock1})', linewidth=1)
    plt.plot(range(len(dates)), prices_2, label=f'资产2 ({stock2})', linewidth=1)
    plt.title(f'配对 {pair_idx}: {stock1}-{stock2} 价格走势')
    plt.legend()
    plt.grid(True, alpha=0.3)
    format_dates(ax1)
    
    # Z-score图
    ax2 = plt.subplot(4, 1, 2)
    plt.plot(range(len(dates)), z_scores, 'b-', linewidth=1)
    plt.axhline(y=entry_threshold, color='r', linestyle='--', label='开仓阈值')
    plt.axhline(y=-entry_threshold, color='r', linestyle='--')
    plt.axhline(y=exit_threshold, color='g', linestyle='--', label='平仓阈值')
    plt.axhline(y=-exit_threshold, color='g', linestyle='--')
    plt.axhline(y=stop_loss, color='m', linestyle='--', label='止损阈值')
    plt.axhline(y=-stop_loss, color='m', linestyle='--')
    plt.title('Z-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    format_dates(ax2)
    
    # 仓位图
    ax3 = plt.subplot(4, 1, 3)
    positions_1 = [p[0] for p in strategy.positions_history]
    positions_2 = [p[1] for p in strategy.positions_history]
    plt.plot(range(len(dates)), positions_1, label='资产1仓位', linewidth=1)
    plt.plot(range(len(dates)), positions_2, label='资产2仓位', linewidth=1)
    plt.title('仓位变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    format_dates(ax3)
    
    # 累计回报图
    ax4 = plt.subplot(4, 1, 4)
    plt.plot(range(len(dates)), cumulative_returns, 'b-', linewidth=1)
    plt.title(f'累计回报 (总回报: {total_reward:.4f}, 夏普比率: {sharpe_ratio:.4f})')
    plt.grid(True, alpha=0.3)
    format_dates(ax4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"pair_{pair_idx}_performance.png"), dpi=300)
    plt.close()

def run_baseline(data_path, pairs_path, results_dir=None):
    """运行Z-score基准策略并评估结果"""
    if results_dir is None:
        results_dir = f"results/baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("加载数据...")
    # 读取数据
    data = pd.read_csv(data_path)
    pairs_data = pd.read_csv(pairs_path)
    
    # 打印原始数据的股票代码样例
    print("\n原始数据中的股票代码样例:")
    print(data['stock_code'].head())
    print("\n配对数据中的股票代码样例:")
    print(pairs_data[['stock1', 'stock2']].head())
    
    # 数据预处理
    # 检查stock_code的数据类型
    print("\n股票代码数据类型:")
    print(f"原始数据 stock_code 类型: {data['stock_code'].dtype}")
    print(f"配对数据 stock1 类型: {pairs_data['stock1'].dtype}")
    print(f"配对数据 stock2 类型: {pairs_data['stock2'].dtype}")
    
    # 统一股票代码格式
    data['stock_code'] = data['stock_code'].astype(str)
    pairs_data['stock1'] = pairs_data['stock1'].astype(str)
    pairs_data['stock2'] = pairs_data['stock2'].astype(str)
    
    # 打印处理后的股票代码
    print("\n处理后的股票代码样例:")
    print("原始数据:")
    print(data['stock_code'].head())
    print("\n配对数据:")
    print(pairs_data[['stock1', 'stock2']].head())
    
    # 获取所有可用的股票代码
    available_stocks = set(data['stock_code'].unique())
    print("\n可用的股票代码:")
    print(sorted(list(available_stocks))[:10])  # 只打印前10个
    
    # 检查数据完整性
    print("\n数据完整性检查:")
    print(f"原始数据行数: {len(data)}")
    print(f"唯一股票数: {len(available_stocks)}")
    print(f"日期范围: {data['date'].min()} 到 {data['date'].max()}")
    
    # 创建价格数据透视表
    price_pivot = data.pivot(index='date', columns='stock_code', values='close')
    print(f"\n价格数据透视表形状: {price_pivot.shape}")
    print("价格数据透视表列名:")
    print(sorted(price_pivot.columns.tolist())[:10])  # 只打印前10个
    
    # 检查配对数据
    print("\n配对数据检查:")
    valid_pairs = []
    for idx, row in pairs_data.iterrows():
        stock1 = str(row['stock1'])
        stock2 = str(row['stock2'])
        
        print(f"\n检查配对 {stock1}-{stock2}:")
        print(f"stock1 在可用股票中: {stock1 in available_stocks}")
        print(f"stock2 在可用股票中: {stock2 in available_stocks}")
        
        if stock1 in available_stocks and stock2 in available_stocks:
            # 检查是否有足够的数据
            data1 = price_pivot[stock1].dropna()
            data2 = price_pivot[stock2].dropna()
            
            print(f"stock1 数据点数: {len(data1)}")
            print(f"stock2 数据点数: {len(data2)}")
            
            if len(data1) > 0 and len(data2) > 0:
                valid_pairs.append((stock1, stock2, row['ratio_mean'], row['ratio_std']))
                print(f"配对有效，共同数据点数: {len(data1)}")
            else:
                print(f"配对 {stock1}-{stock2} 数据不足")
        else:
            print(f"配对 {stock1}-{stock2} 股票代码不存在于价格数据中")
    
    print(f"\n有效配对数量: {len(valid_pairs)}/{len(pairs_data)}")
    
    if len(valid_pairs) == 0:
        print("错误：没有找到有效的配对！")
        return None
    
    # 评估结果存储
    all_returns = []
    pair_returns = {}
    all_sharpe_ratios = []
    all_max_drawdowns = []
    
    # 对每个有效配对评估策略
    print("\n开始评估基准策略...")
    for pair_idx, (stock1, stock2, ratio_mean, ratio_std) in enumerate(valid_pairs, 1):
        print(f"\n处理配对 {pair_idx}: {stock1}-{stock2}")
        
        # 获取价格数据
        pair_data = price_pivot[[stock1, stock2]].dropna()
        
        if len(pair_data) == 0:
            print(f"警告: 配对 {stock1}-{stock2} 没有共同的交易日期")
            continue
            
        # 计算价格比率和z-score
        prices_1 = pair_data[stock1].values
        prices_2 = pair_data[stock2].values
        dates = pair_data.index.values
        
        price_ratio = prices_1 / prices_2
        z_scores = (price_ratio - ratio_mean) / ratio_std
        
        print(f"数据点数: {len(pair_data)}")
        print(f"价格比率统计: 均值={np.mean(price_ratio):.4f}, 标准差={np.std(price_ratio):.4f}")
        print(f"Z-score范围: [{np.min(z_scores):.4f}, {np.max(z_scores):.4f}]")
        
        # 初始化策略
        strategy = ZScoreStrategy(
            entry_threshold=1.0,
            exit_threshold=0.0,
            stop_loss=2.5,
            buy_cost_rate=0.00032,
            sell_cost_rate=0.00132
        )
        
        # 遍历每个交易日
        daily_returns = []
        cumulative_returns = []
        total_reward = 0
        
        for t in range(len(dates)):
            price_1 = prices_1[t]
            price_2 = prices_2[t]
            z_score = z_scores[t]
            
            # 决定并执行交易
            action = strategy.decide_action(z_score, price_ratio[t])
            reward = strategy.execute_action(action, price_1, price_2, z_score)
            total_reward += reward
            daily_returns.append(reward)
            cumulative_returns.append(total_reward)
        
        # 计算统计指标
        if len(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
            max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
            
            all_returns.append(total_reward)
            pair_returns[f"配对_{pair_idx}"] = total_reward
            all_sharpe_ratios.append(sharpe_ratio)
            all_max_drawdowns.append(max_drawdown)
            
            print(f"\n配对 {pair_idx} 统计:")
            print(f"总回报: {total_reward:.4f}")
            print(f"夏普比率: {sharpe_ratio:.4f}")
            print(f"最大回撤: {max_drawdown:.4f}")
            
            # 绘制结果图表
            plot_pair_performance(
                dates=dates,
                prices_1=prices_1,
                prices_2=prices_2,
                z_scores=z_scores,
                strategy=strategy,
                cumulative_returns=cumulative_returns,
                pair_idx=pair_idx,
                stock1=stock1,
                stock2=stock2,
                total_reward=total_reward,
                sharpe_ratio=sharpe_ratio,
                results_dir=results_dir,
                entry_threshold=1.0,
                exit_threshold=0.0,
                stop_loss=2.5
            )
    
    # 计算整体统计数据
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    overall_sharpe = mean_return / (std_return + 1e-10)
    positive_pairs = sum(1 for r in all_returns if r > 0)
    negative_pairs = sum(1 for r in all_returns if r <= 0)
    
    # 绘制所有配对的回报比较
    plt.figure(figsize=(12, 6))
    pairs = list(pair_returns.keys())
    returns = list(pair_returns.values())
    plt.bar(pairs, returns)
    plt.title('各配对回报比较 (基准策略)')
    plt.ylabel('总回报')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_pairs_returns.png"))
    plt.close()
    
    # 绘制回报分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(all_returns, bins=10, alpha=0.7, color='blue')
    plt.axvline(x=mean_return, color='r', linestyle='--', label=f'平均值: {mean_return:.4f}')
    plt.title('回报分布 (基准策略)')
    plt.xlabel('回报')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "returns_distribution.png"))
    plt.close()
    
    # 绘制夏普比率比较
    plt.figure(figsize=(12, 6))
    plt.bar(pairs, all_sharpe_ratios)
    plt.title('各配对夏普比率比较 (基准策略)')
    plt.ylabel('夏普比率')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_pairs_sharpe.png"))
    plt.close()
    
    # 绘制最大回撤比较
    plt.figure(figsize=(12, 6))
    plt.bar(pairs, all_max_drawdowns)
    plt.title('各配对最大回撤比较 (基准策略)')
    plt.ylabel('最大回撤')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "max_drawdowns.png"))
    plt.close()
    
    # 创建详细的评估报告
    with open(os.path.join(results_dir, "baseline_report.txt"), "w") as f:
        f.write("Z-Score基准策略评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("策略参数:\n")
        f.write(f"开仓阈值: {1.0}\n")
        f.write(f"平仓阈值: {0.0}\n")
        f.write(f"止损阈值: {2.5}\n")
        f.write(f"交易成本: {0.0003}\n\n")
        
        f.write("整体性能指标:\n")
        f.write(f"平均回报: {mean_return:.4f}\n")
        f.write(f"回报标准差: {std_return:.4f}\n")
        f.write(f"整体夏普比率: {overall_sharpe:.4f}\n")
        f.write(f"盈利配对数: {positive_pairs}/{len(all_returns)}\n")
        f.write(f"亏损配对数: {negative_pairs}/{len(all_returns)}\n")
        f.write(f"平均最大回撤: {np.mean(all_max_drawdowns):.4f}\n\n")
        
        f.write("各配对回报:\n")
        for pair, ret in pair_returns.items():
            f.write(f"{pair}: {ret:.4f}\n")
    
    print("基准策略评估完成!")
    print(f"评估报告保存在: {os.path.join(results_dir, 'baseline_report.txt')}")
    
    return {
        'mean_return': mean_return,
        'sharpe_ratio': overall_sharpe,
        'positive_pairs': positive_pairs,
        'total_pairs': len(all_returns),
        'max_drawdown': np.mean(all_max_drawdowns)
    }

if __name__ == "__main__":
    # 运行基准策略
    data_path = 'data/processed_hedge_data.csv'
    pairs_path = 'data/hedge_pairs.csv'
    
    results = run_baseline(data_path, pairs_path)
    print(f"基准策略平均回报: {results['mean_return']:.4f}")
    print(f"基准策略夏普比率: {results['sharpe_ratio']:.4f}")
    print(f"盈利配对比例: {results['positive_pairs']}/{results['total_pairs']}")