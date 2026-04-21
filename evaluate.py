import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from models.dqn_model import HybridAgent
from models.hedge_env import HedgeEnv

def calculate_max_drawdown(returns):
    # 计算累计收益
    cumulative_returns = np.cumsum(returns)
    # 计算当前高点
    running_max = np.maximum.accumulate(cumulative_returns)
    # 计算回撤
    drawdown = running_max - cumulative_returns
    # 最大回撤
    max_drawdown = np.max(drawdown)
    return max_drawdown

def evaluate_model():
    # 设置matplotlib参数
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    plt.rcParams['font.size'] = 12  # 设置全局字体大小
    plt.rcParams['axes.titlesize'] = 14  # 设置标题字体大小
    plt.rcParams['figure.titlesize'] = 16  # 设置图表标题字体大小
    plt.rcParams['figure.figsize'] = [15, 12]  # 设置默认图表大小
    
    # 创建保存目录
    results_dir = 'results/train'
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载数据和模型
    data = pd.read_csv('data/processed_hedge_data.csv')
    pairs_data = pd.read_csv('data/hedge_pairs.csv')
    
    # 统一股票代码格式
    data['stock_code'] = data['stock_code'].astype(float).astype(str)
    pairs_data['stock1'] = pairs_data['stock1'].astype(float).astype(str)
    pairs_data['stock2'] = pairs_data['stock2'].astype(float).astype(str)
    
    # 创建环境和加载模型
    env = HedgeEnv(data, pairs_data)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = HybridAgent(state_dim, action_dim)
    agent.load('results/train/best_model.pth')
    
    # 创建价格数据透视表
    price_pivot = data.pivot(index='date', columns='stock_code', values='close')
    
    # 评估结果存储
    all_returns = []
    pair_returns = {}
    all_sharpe_ratios = []
    all_max_drawdowns = []
    
    # 对每个配对进行评估
    for pair_idx, pair in enumerate(pairs_data.iterrows(), 1):
        stock1 = pair[1]['stock1']
        stock2 = pair[1]['stock2']
        print(f"\n评估配对: {stock1}-{stock2}")
        
        # 获取价格数据
        pair_data = price_pivot[[stock1, stock2]].dropna()
        prices_1 = pair_data[stock1].values
        prices_2 = pair_data[stock2].values
        dates = pair_data.index.values
        
        # 计算价格比率和z-score
        price_ratio = prices_1 / prices_2
        z_scores = (price_ratio - pair[1]['ratio_mean']) / pair[1]['ratio_std']
        
        # 运行模型获取仓位和回报
        env.current_pair_idx = pair_idx - 1
        agent.reset()   # 重置时序缓冲区
        state = env.reset()
        done = False
        positions = []
        returns = []
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            positions.append(env.position)
            returns.append(reward)
            state = next_state
        

            
        # 计算夏普比率
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        # 绘制四子图
        plt.figure(figsize=(15, 12))
        
        # 价格图
        ax1 = plt.subplot(4, 1, 1)
        plt.plot(prices_1, label=f'资产1 ({stock1})', linewidth=1)
        plt.plot(prices_2, label=f'资产2 ({stock2})', linewidth=1)
        plt.title(f'配对 {pair_idx}: {stock1}-{stock2} 价格走势')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Z-score图
        ax2 = plt.subplot(4, 1, 2)
        plt.plot(z_scores, 'b-', linewidth=1)
        plt.axhline(y=1.0, color='r', linestyle='--', label='开仓阈值')
        plt.axhline(y=-1.0, color='r', linestyle='--')
        plt.axhline(y=0.0, color='g', linestyle='--', label='平仓阈值')
        plt.axhline(y=2.5, color='m', linestyle='--', label='止损阈值')
        plt.axhline(y=-2.5, color='m', linestyle='--')
        plt.title('Z-Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 仓位图
        ax3 = plt.subplot(4, 1, 3)
        positions = np.array(positions)
        plt.plot(positions[:, 0], label='资产1仓位', linewidth=1)
        plt.plot(positions[:, 1], label='资产2仓位', linewidth=1)
        plt.title('仓位变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 累计回报图
        ax4 = plt.subplot(4, 1, 4)
        plt.plot(np.cumsum(returns), 'b-', linewidth=1)
        plt.title(f'累计回报 (总回报: {np.sum(returns):.4f}, 夏普比率: {sharpe_ratio:.4f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'pair_{pair_idx}_performance.png'), dpi=300)
        plt.close()
        
        # 存储单个配对的回报
        pair_returns[f"配对_{pair_idx}"] = np.sum(returns)
        all_returns.append(np.sum(returns))
        all_sharpe_ratios.append(sharpe_ratio)
        #计算最大回撤
        max_drawdown = calculate_max_drawdown(returns)
        all_max_drawdowns.append(max_drawdown)

    
    # 计算整体统计数据
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    overall_sharpe = mean_return / (std_return + 1e-10)
    positive_pairs = sum(1 for r in all_returns if r > 0)
    
    # 1. 绘制所有配对的回报比较（柱状图）
    plt.figure(figsize=(12, 6))
    pairs = [f"配对_{i+1}" for i in range(len(all_returns))]
    plt.bar(pairs, all_returns)
    plt.title('各配对回报比较 (强化学习策略)')
    plt.ylabel('总回报')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_pairs_returns_bar.png"))
    plt.close()
    
    # 2. 绘制回报分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(all_returns, bins=10, alpha=0.7, color='blue')
    plt.axvline(x=mean_return, color='r', linestyle='--', 
                label=f'平均值: {mean_return:.4f}')
    plt.title('回报分布 (强化学习策略)')
    plt.xlabel('回报')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "returns_distribution.png"))
    plt.close()
    
    # 3. 绘制夏普比率比较
    plt.figure(figsize=(12, 6))
    plt.bar(pairs, all_sharpe_ratios)
    plt.title('各配对夏普比率比较 (强化学习策略)')
    plt.ylabel('夏普比率')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_pairs_sharpe.png"))
    plt.close()
    
    # 4. 绘制最大回撤比较
    plt.figure(figsize=(12, 6))
    plt.bar(pairs, all_max_drawdowns)
    plt.title('各配对最大回撤比较 (强化学习策略)')
    plt.ylabel('最大回撤')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "max_drawdowns.png"))
    plt.close()

    # 在绘制完所有图表后，添加报告生成代码
    report_path = os.path.join(results_dir, "rl_evaluation_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("强化学习策略评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("策略参数:\n")
        f.write("开仓阈值: 1.0\n")
        f.write("平仓阈值: 0.0\n")
        f.write("止损阈值: 2.5\n")
        f.write("交易成本: 0.0003\n\n")
        
        f.write("整体性能指标:\n")
        f.write(f"平均回报: {mean_return:.4f}\n")
        f.write(f"回报标准差: {std_return:.4f}\n")
        f.write(f"整体夏普比率: {overall_sharpe:.4f}\n")
        f.write(f"盈利配对数: {positive_pairs}/{len(all_returns)}\n")
        f.write(f"亏损配对数: {len(all_returns)-positive_pairs}/{len(all_returns)}\n")
        f.write(f"平均最大回撤: {np.mean(all_max_drawdowns):.4f}\n\n")
        
        f.write("各配对回报:\n")
        for pair, ret in pair_returns.items():
            f.write(f"{pair}: {ret:.4f}\n")
    
    print(f"\n评估报告已保存至: {report_path}")

if __name__ == "__main__":
    evaluate_model() 