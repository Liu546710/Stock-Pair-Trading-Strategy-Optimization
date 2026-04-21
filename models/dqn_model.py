"""
DQN + Actor-Critic 混合深度强化学习模型
========================================
核心改进：
  1. 共享 LSTM 编码器：特征提取 + 时序建模
  2. 市场状态分类器：均值回归 / 趋势 / 震荡 三类市场状态识别
  3. Actor-Critic 分层决策：Actor 输出策略分布，Critic 输出 Q 值
  4. 双重 Q 学习（Double DQN）：减少 Q 值高估偏差
  5. 优先经验回放（PER + Sum-Tree）：提升高价值样本利用率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


# ─────────────────────────────────────────────────────────────
# 1. Sum-Tree：支持 O(log n) 优先级采样
# ─────────────────────────────────────────────────────────────

class SumTree:
    """
    Sum-Tree 叶节点存储转移样本，内部节点存储子树优先级之和。
    支持 O(log n) 的插入与采样。
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ─────────────────────────────────────────────────────────────
# 2. 优先经验回放缓冲区（PER）
# ─────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    基于 Sum-Tree 的优先经验回放。
    alpha: 优先级指数（0=均匀, 1=完全优先）
    beta:  重要性采样修正指数（线性退火到 1.0）
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.eps = 1e-6

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + self.frame *
                   (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *transition):
        max_priority = self.tree.tree[self.tree.capacity - 1:].max()
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, transition)

    def sample(self, batch_size: int):
        batch, indices, priorities = [], [], []
        segment = self.tree.total / batch_size
        self.frame += 1

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data == 0:
                continue
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        sampling_probs = np.array(priorities) / (self.tree.total + self.eps)
        is_weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        is_weights /= is_weights.max()
        return batch, indices, is_weights

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.eps) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ─────────────────────────────────────────────────────────────
# 3. 共享 LSTM 编码器
# ─────────────────────────────────────────────────────────────

class SharedEncoder(nn.Module):
    """
    输入：[batch, seq_len, state_dim]
    输出：[batch, hidden_dim] — 时序上下文特征
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.layer_norm(out[:, -1, :])


# ─────────────────────────────────────────────────────────────
# 4. 市场状态分类器
# ─────────────────────────────────────────────────────────────

class MarketStateClassifier(nn.Module):
    """
    将 LSTM 特征映射到市场状态：
        0 → 均值回归（mean reversion）
        1 → 趋势     （trend）
        2 → 震荡     （oscillation）
    """
    def __init__(self, hidden_dim: int = 64, num_states: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_states),
        )

    def forward(self, features: torch.Tensor):
        return self.net(features)


# ─────────────────────────────────────────────────────────────
# 5. Actor（策略网络）
# ─────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """输入：LSTM 特征 + 市场状态嵌入 → 动作概率分布"""
    def __init__(self, context_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, action_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(context), dim=-1)


# ─────────────────────────────────────────────────────────────
# 6. Critic（价值网络 / DQN）
# ─────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """输入：LSTM 特征 + 市场状态嵌入 → 每个动作的 Q 值"""
    def __init__(self, context_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, action_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


# ─────────────────────────────────────────────────────────────
# 7. 混合网络
# ─────────────────────────────────────────────────────────────

class HybridNetwork(nn.Module):
    """
    架构：SharedEncoder → MarketStateClassifier
                        ↘ Actor
                        ↘ Critic
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 64, num_states: int = 3):
        super().__init__()
        self.encoder = SharedEncoder(state_dim, hidden_dim)
        self.market_classifier = MarketStateClassifier(hidden_dim, num_states)
        context_dim = hidden_dim + num_states
        self.actor = Actor(context_dim, action_dim)
        self.critic = Critic(context_dim, action_dim)

    def forward(self, x: torch.Tensor):
        """
        x: [batch, seq_len, state_dim]
        Returns:
            q_values:      [batch, action_dim]
            action_probs:  [batch, action_dim]
            market_logits: [batch, num_states]
        """
        features = self.encoder(x)
        market_logits = self.market_classifier(features)
        market_embed = F.softmax(market_logits, dim=-1)
        context = torch.cat([features, market_embed], dim=-1)
        return self.critic(context), self.actor(context), market_logits


# ─────────────────────────────────────────────────────────────
# 辅助：市场状态伪标签推导
# ─────────────────────────────────────────────────────────────

def derive_market_labels(states: torch.Tensor) -> torch.Tensor:
    """
    根据状态特征推导市场状态伪标签。
    states: [batch, seq_len, state_dim]，取最后时间步
    特征布局（对应 HedgeEnv._get_observation）：
        [0]=z_score  [1]=z_velocity  [2]=z_volatility ...
    标签：0=均值回归, 1=趋势, 2=震荡
    """
    last = states[:, -1, :]
    z_velocity = last[:, 1]
    z_volatility = last[:, 2]

    labels = torch.zeros(states.size(0), dtype=torch.long, device=states.device)
    trend_mask = torch.abs(z_velocity) > 0.3
    labels[trend_mask] = 1
    oscillation_mask = (z_volatility > 1.0) & ~trend_mask
    labels[oscillation_mask] = 2
    return labels


# ─────────────────────────────────────────────────────────────
# 8. 混合智能体（HybridAgent）
# ─────────────────────────────────────────────────────────────

class HybridAgent:
    """
    DQN + Actor-Critic 混合智能体。

    决策层：
        action = argmax(λ·softmax(Q) + (1-λ)·π)
        λ 根据市场状态动态调整

    学习层：
        L = L_Q(Double DQN) + α_π·L_π + α_m·L_m + α_H·L_H
    """

    # 市场状态对应的 Q/π 融合权重
    MARKET_LAMBDA = {0: 0.75, 1: 0.60, 2: 0.50}

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        hidden_dim: int = 64,
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.98,
        memory_size: int = 50_000,
        batch_size: int = 64,
        tau: float = 0.005,
        alpha_actor: float = 0.5,
        alpha_market: float = 0.3,
        alpha_entropy: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.alpha_actor = alpha_actor
        self.alpha_market = alpha_market
        self.alpha_entropy = alpha_entropy

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_network = HybridNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = HybridNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate
        )
        self.memory = PrioritizedReplayBuffer(memory_size)

        self.state_buffer = deque(maxlen=seq_len)
        self._current_state_seq = None
        self.reset()

    # ── 工具 ─────────────────────────────────────────────────

    def reset(self):
        """每个 episode 开始时重置观测序列缓冲区"""
        for _ in range(self.seq_len):
            self.state_buffer.append(np.zeros(self.state_dim, dtype=np.float32))
        self._current_state_seq = None

    def _get_state_seq(self, state: np.ndarray) -> np.ndarray:
        self.state_buffer.append(state.astype(np.float32))
        return np.array(list(self.state_buffer), dtype=np.float32)

    # ── 动作选择 ─────────────────────────────────────────────

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_seq = self._get_state_seq(state)
        self._current_state_seq = state_seq

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            x = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            q_values, action_probs, market_logits = self.online_network(x)
            market_state = market_logits.argmax(dim=-1).item()
            lam = self.MARKET_LAMBDA.get(market_state, 0.65)
            combined = lam * F.softmax(q_values, dim=-1) + (1 - lam) * action_probs
            return combined.argmax(dim=-1).item()

    # ── 经验存储 ─────────────────────────────────────────────

    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, next_state: np.ndarray, done: bool):
        state_seq = self._current_state_seq
        temp_buf = deque(list(self.state_buffer), maxlen=self.seq_len)
        temp_buf.append(next_state.astype(np.float32))
        next_state_seq = np.array(list(temp_buf), dtype=np.float32)
        self.memory.push(state_seq, action, reward, next_state_seq, float(done))

    # ── 网络更新 ─────────────────────────────────────────────

    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch, tree_indices, is_weights = self.memory.sample(self.batch_size)
        if not batch:
            return 0.0

        state_seqs, actions, rewards, next_state_seqs, dones = zip(*batch)

        state_t  = torch.FloatTensor(np.array(state_seqs)).to(self.device)
        next_t   = torch.FloatTensor(np.array(next_state_seqs)).to(self.device)
        action_t = torch.LongTensor(actions).to(self.device)
        reward_t = torch.FloatTensor(rewards).to(self.device)
        done_t   = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(is_weights).to(self.device)

        # ── Double DQN 目标 Q 值 ─────────────────────────────
        with torch.no_grad():
            next_q_online, _, _ = self.online_network(next_t)
            best_actions = next_q_online.argmax(dim=-1, keepdim=True)
            next_q_target, _, _ = self.target_network(next_t)
            next_q = next_q_target.gather(1, best_actions).squeeze(1)
            target_q = reward_t + (1 - done_t) * self.gamma * next_q

        # ── 在线网络前向 ─────────────────────────────────────
        q_values, action_probs, market_logits = self.online_network(state_t)
        current_q = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

        # ── Critic 损失（加权 Huber）────────────────────────
        td_errors = (target_q - current_q).detach().cpu().numpy()
        critic_loss = (weights_t * F.smooth_l1_loss(
            current_q, target_q, reduction='none')).mean()

        # ── Actor 损失（标准策略梯度：只用实际执行动作）────────
        with torch.no_grad():
            state_value = (action_probs.detach() * q_values.detach()).sum(dim=-1, keepdim=True)  # [B,1]
            advantage = q_values.detach() - state_value   # [B, action_dim]
        selected_log_probs = torch.log(
            action_probs.gather(1, action_t.unsqueeze(1)).squeeze(1) + 1e-8
        )  # [B]
        selected_advantage = advantage.gather(1, action_t.unsqueeze(1)).squeeze(1)  # [B]
        actor_loss = -(selected_log_probs * selected_advantage).mean()

        # ── 市场状态分类损失（伪标签）───────────────────────
        market_labels = derive_market_labels(state_t)
        market_loss = F.cross_entropy(market_logits, market_labels)

        # ── 策略熵正则化 ─────────────────────────────────────
        log_probs = torch.log(action_probs + 1e-8)
        entropy_loss = (action_probs * log_probs).sum(dim=-1).mean()  # 最小化负熵=最大化熵

        loss = (critic_loss
                + self.alpha_actor * actor_loss
                + self.alpha_market * market_loss
                + self.alpha_entropy * entropy_loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 1.0)
        self.optimizer.step()

        self.memory.update_priorities(tree_indices, td_errors)
        self._soft_update()

        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def _soft_update(self):
        for t_p, o_p in zip(self.target_network.parameters(),
                             self.online_network.parameters()):
            t_p.data.copy_(t_p.data * (1 - self.tau) + o_p.data * self.tau)

    # ── 保存 / 加载 ──────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            'online_state_dict': self.online_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'seq_len': self.seq_len,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        # 如果存储的 seq_len 与当前不一致，重建缓冲区
        saved_seq_len = checkpoint.get('seq_len', self.seq_len)
        if saved_seq_len != self.seq_len:
            self.seq_len = saved_seq_len
            self.state_buffer = deque(maxlen=self.seq_len)
            self.reset()
        self.online_network.load_state_dict(checkpoint['online_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)


# ── 向后兼容别名 ─────────────────────────────────────────────
DQNAgent = HybridAgent
