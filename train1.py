import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.hedge_env import HedgeEnv
from models.dqn_model import HybridAgent
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

class RLTrainer:
    def __init__(self, data_path, pairs_path, results_dir='results/train'):
        self.data_path = data_path
        self.pairs_path = pairs_path
        self.results_dir = results_dir

        # 创建结果目录
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # 训练参数
        self.episodes = 200       # 减少轮数
        self.eval_interval = 5    # 更频繁评估

        # 添加早停机制
        self.patience = 10         # 适中的早停
        self.best_return = -np.inf
        self.no_improve = 0
        self.eval_history = []    # 添加评估历史记录

    def train(self):
        """训练DRL模型"""
        print("加载数据...")
        data = pd.read_csv(self.data_path)
        pairs_data = pd.read_csv(self.pairs_path)

        # ── 按时间切分：训练集70% / 验证集15% / 测试集15% ──
        data['date'] = pd.to_datetime(data['date'])
        dates = sorted(data['date'].unique())
        n = len(dates)
        train_end = dates[int(n * 0.70)]
        val_end   = dates[int(n * 0.85)]

        train_data = data[data['date'] <= train_end].copy()
        val_data   = data[(data['date'] > train_end) & (data['date'] <= val_end)].copy()
        # 测试集留给 evaluate.py 使用，这里只保存切分点供参考
        print(f"训练集: {dates[0].date()} ~ {train_end.date()} ({int(n*0.70)} 个交易日)")
        print(f"验证集: {dates[int(n*0.70)+1].date()} ~ {val_end.date()} ({int(n*0.15)} 个交易日)")
        print(f"测试集: {dates[int(n*0.85)+1].date()} ~ {dates[-1].date()} ({n - int(n*0.85) - 1} 个交易日)")

        # 保存切分点，供 evaluate.py 读取
        split_info = {
            'train_end': str(train_end.date()),
            'val_end':   str(val_end.date()),
        }
        import json
        with open(os.path.join(self.results_dir, 'data_split.json'), 'w') as f:
            json.dump(split_info, f, indent=2)

        # 训练环境使用训练集，验证环境使用验证集
        train_env = HedgeEnv(train_data, pairs_data)
        val_env   = HedgeEnv(val_data,   pairs_data)

        # 创建混合智能体
        state_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.n
        agent = HybridAgent(state_dim, action_dim)

        print("\n开始训练...")
        for episode in tqdm(range(self.episodes)):
            agent.reset()   # 重置时序缓冲区
            state = train_env.reset()
            episode_reward = 0
            done = False

            # 单个episode的训练循环
            while not done:
                # 选择动作
                action = agent.select_action(state)

                # 执行动作
                next_state, reward, done, info = train_env.step(action)

                # 存储经验
                agent.store_transition(state, action, reward, next_state, done)

                # 更新网络
                loss = agent.update()

                state = next_state
                episode_reward += reward

            # 定期评估（使用验证集，与训练集时间段隔离）
            if (episode + 1) % self.eval_interval == 0:
                eval_return = self.evaluate(val_env, agent)
                self.eval_history.append(eval_return)

                print(f"\nEpisode {episode + 1}")
                print(f"评估回报: {eval_return:.4f}")
                print(f"训练回报: {episode_reward:.4f}")
                print(f"Epsilon: {agent.epsilon:.4f}")

                # 检查是否有改善
                if eval_return > self.best_return:
                    print(f"发现更好的模型! 旧的最佳回报: {self.best_return:.4f}, 新的最佳回报: {eval_return:.4f}")
                    self.best_return = eval_return
                    agent.save(os.path.join(self.results_dir, 'best_model.pth'))
                    self.no_improve = 0
                else:
                    self.no_improve += 1
                    print(f"未改善轮数: {self.no_improve}/{self.patience}")

                # 早停检查
                if self.no_improve >= self.patience:
                    print(f"\n触发早停! {self.patience}轮未见改善")
                    print(f"最佳评估回报: {self.best_return:.4f}")
                    break

        print("\n训练完成!")
        return self.best_return

    def evaluate(self, env, agent):
        """评估当前模型"""
        agent.reset()   # 重置时序缓冲区，避免训练状态污染评估
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        return total_reward

if __name__ == "__main__":
    # 运行训练
    trainer = RLTrainer(
        data_path='data/processed_hedge_data.csv',
        pairs_path='data/hedge_pairs.csv'
    )
    best_return = trainer.train()
    print(f"最佳评估回报: {best_return:.4f}")