import numpy as np
import pandas as pd

# 轻量级 spaces 替代，无需安装 gym
class _Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low; self.high = high; self.shape = shape; self.dtype = dtype
class _Discrete:
    def __init__(self, n):
        self.n = n
class spaces:
    Box = _Box
    Discrete = _Discrete

class HedgeEnv:
    def __init__(self, data, pairs_data):
        # 保存配对数据
        self.pairs_data = pairs_data
        self.current_pair_idx = -1
        self.data = data
        
        # 设置观察和动作空间
        self.window_size = 20
        # 8维状态：z_score, z_velocity, z_volatility, price_ratio_norm,
        #          position[0], position[1], total_return_norm, time_progress
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 买入、卖出、持有
        
        # 初始化资金相关属性
        self.initial_capital = 1_000_000.0   # 初始资金100万
        self.cash = self.initial_capital
        # 手续费模型（A股实际场景，忽略个人做空限制）
        #   买入方：经纪佣金 0.03% + 过户费 0.002% ≈ 0.032%
        #   卖出方：经纪佣金 0.03% + 过户费 0.002% + 印花税 0.1% ≈ 0.132%
        self.buy_cost_rate  = 0.00032   # 买入单边费率
        self.sell_cost_rate = 0.00132   # 卖出单边费率（含印花税）
        
        # 初始化收益相关属性
        self.total_return = 0.0
        self.returns_history = []
        self.position_value = 0.0
        
        self.reset()
        
    def reset(self):
        """重置环境状态"""
        # 更新当前配对索引
        self.current_pair_idx = (self.current_pair_idx + 1) % len(self.pairs_data)
        
        # 获取当前配对的股票代码
        self.stock1 = self.pairs_data.iloc[self.current_pair_idx]['stock1']
        self.stock2 = self.pairs_data.iloc[self.current_pair_idx]['stock2']
        
        # 获取两只股票的价格数据
        stock1_data = self.data[self.data['stock_code'] == self.stock1]
        stock2_data = self.data[self.data['stock_code'] == self.stock2]
        
        # 直接使用价格数据
        self.stock1_prices = stock1_data['close'].values
        self.stock2_prices = stock2_data['close'].values
        
        # 预计算滚动窗口均値和标准差，避免前视偏差
        # 只使用截至当前时刻的历史数据，不包含未来信息
        price_ratios = pd.Series(self.stock1_prices / self.stock2_prices)
        self.rolling_ratio_mean = (
            price_ratios.rolling(self.window_size, min_periods=2).mean().values
        )
        self.rolling_ratio_std = (
            price_ratios.rolling(self.window_size, min_periods=2).std(ddof=1).values
        )
        
        # 重置状态
        self.current_step = self.window_size
        self.position = [0, 0]  # 初始持仓为0
        self.cash = self.initial_capital
        self.total_return = 0.0
        self.returns_history = []
        self.position_value = 0.0
        
        # 设置当前价格
        self.current_price_1 = self.stock1_prices[self.current_step]
        self.current_price_2 = self.stock2_prices[self.current_step]
        
        return self._get_observation()
        
    def step(self, action):
        """执行一步交易"""
        # 保存旧状态
        old_pos_1 = self.position[0]
        old_pos_2 = self.position[1]
        old_value = self.calculate_portfolio_value()
        
        # 更新价格
        self.current_price_1 = self.stock1_prices[self.current_step]
        self.current_price_2 = self.stock2_prices[self.current_step]
        
        # 计算z-score
        z_score = self._calculate_z_score()
        
        # 执行交易
        reward = 0
        if action == 1:  # 买入资产1（多头），卖出资产2（空头）
            if self.position[0] <= 0:
                # 先平旧空仓（买入平仓，买入费率）
                if self.position[0] < 0:
                    reward -= self.buy_cost_rate * self.current_price_1
                if self.position[1] > 0:
                    reward -= self.sell_cost_rate * self.current_price_2
                # 开新仓（买asset1 + 卖asset2）
                self.position[0] = 1
                self.position[1] = -1
                reward -= (self.buy_cost_rate  * self.current_price_1 +
                           self.sell_cost_rate * self.current_price_2)

        elif action == 2:  # 卖出资产1（空头），买入资产2（多头）
            if self.position[0] >= 0:
                # 先平旧多仓（卖出平仓，卖出费率）
                if self.position[0] > 0:
                    reward -= self.sell_cost_rate * self.current_price_1
                if self.position[1] < 0:
                    reward -= self.buy_cost_rate * self.current_price_2
                # 开新仓（卖asset1 + 买asset2）
                self.position[0] = -1
                self.position[1] = 1
                reward -= (self.sell_cost_rate * self.current_price_1 +
                           self.buy_cost_rate  * self.current_price_2)
                
        # 计算持仓收益
        if len(self.position) > 0:
            if self.position[0] != 0:
                reward += self.position[0] * (self.current_price_1 - self.stock1_prices[self.current_step - 1])
            if self.position[1] != 0:
                reward += self.position[1] * (self.current_price_2 - self.stock2_prices[self.current_step - 1])
        
        # 更新总收益
        self.total_return += reward
        
        # 更新步数
        self.current_step += 1

        # 检查是否结束
        done = self.current_step >= len(self.stock1_prices) - 1

        # episode 结束时强制平仓，结算未实现益亏
        if done and (self.position[0] != 0 or self.position[1] != 0):
            close_pnl = (self.position[0] * self.current_price_1 +
                         self.position[1] * self.current_price_2)
            # 平仓费用：正持仓卖出（卖出费率），负持仓买入平仓（买入费率）
            cost = 0.0
            if self.position[0] > 0:
                cost += self.sell_cost_rate * self.current_price_1
            elif self.position[0] < 0:
                cost += self.buy_cost_rate * self.current_price_1
            if self.position[1] > 0:
                cost += self.sell_cost_rate * self.current_price_2
            elif self.position[1] < 0:
                cost += self.buy_cost_rate * self.current_price_2
            close_pnl -= cost
            reward += close_pnl
            self.total_return += close_pnl
            self.position = [0, 0]
        
        return self._get_observation(), self.calculate_reward(reward, old_pos_1 != self.position[0] or old_pos_2 != self.position[1]), done, {
            'position_1': self.position[0],
            'position_2': self.position[1],
            'z_score': z_score,
            'total_return': self.total_return
        }
        
    def _calculate_z_score(self):
        """计算当前价格比率的z-score（滚动窗口，无前视偏差）"""
        ratio_mean = self.rolling_ratio_mean[self.current_step]
        ratio_std  = self.rolling_ratio_std[self.current_step]
        if np.isnan(ratio_mean) or np.isnan(ratio_std) or ratio_std == 0:
            return 0.0
        price_ratio = self.current_price_1 / self.current_price_2
        return (price_ratio - ratio_mean) / ratio_std
        
    def _get_observation(self):
        """获取当前状态观察（8维）"""
        z_score = self._calculate_z_score()

        # z-score 变化速度
        if self.current_step > self.window_size:
            prev_ratio = (self.stock1_prices[self.current_step - 1] /
                          self.stock2_prices[self.current_step - 1])
            prev_mean = self.rolling_ratio_mean[self.current_step - 1]
            prev_std  = self.rolling_ratio_std[self.current_step - 1]
            if not (np.isnan(prev_mean) or np.isnan(prev_std) or prev_std == 0):
                prev_z = (prev_ratio - prev_mean) / prev_std
                z_velocity = float(np.clip(z_score - prev_z, -2.0, 2.0))
            else:
                z_velocity = 0.0
        else:
            z_velocity = 0.0

        # 近期 z-score 波动率（滚助10步标准差）
        start = max(0, self.current_step - 10)
        recent_ratios = (self.stock1_prices[start:self.current_step] /
                         self.stock2_prices[start:self.current_step])
        recent_means = self.rolling_ratio_mean[start:self.current_step]
        recent_stds  = self.rolling_ratio_std[start:self.current_step]
        valid = ~(np.isnan(recent_means) | np.isnan(recent_stds) | (recent_stds == 0))
        if valid.sum() > 1:
            recent_z = (recent_ratios[valid] - recent_means[valid]) / recent_stds[valid]
            z_volatility = float(np.clip(np.std(recent_z), 0.0, 3.0))
        else:
            z_volatility = 0.0

        # 未实现盆亏率（按初始资金归一化）
        unrealized_pnl = (self.position[0] * self.current_price_1 +
                          self.position[1] * self.current_price_2)
        unrealized_pnl_norm = float(np.clip(
            unrealized_pnl / self.initial_capital * 100, -10.0, 10.0))

        # 归一化累计收益
        total_return_norm = float(np.clip(
            self.total_return / self.initial_capital * 100, -10.0, 10.0))

        # 时间进度 [0, 1]
        time_progress = float(self.current_step) / max(len(self.stock1_prices), 1)

        return np.array([
            np.clip(z_score, -5.0, 5.0),
            z_velocity,
            z_volatility,
            unrealized_pnl_norm,   # 未实现盆亏率（替代冗余的price_ratio_norm）
            float(self.position[0]),
            float(self.position[1]),
            total_return_norm,
            time_progress,
        ], dtype=np.float32)
        
    def calculate_portfolio_value(self):
        """计算当前组合价值"""
        stock_value = (self.position[0] * self.current_price_1 + 
                      self.position[1] * self.current_price_2)
        return self.cash + stock_value

    def calculate_reward(self, returns, position_changed):
        """优化的奖励函数"""
        reward = returns * 100
        z_score = self._calculate_z_score()
        
        # 模仿基准模型的开平仓逻辑
        if self.position[0] == 0:  # 无仓位
            if abs(z_score) > 1.0:  # 基准模型的开仓阈值
                reward *= 2.0  # 大幅奖励正确的开仓时机
        else:  # 有仓位
            if abs(z_score) < 0.2:  # 回归均值
                reward *= 1.5  # 奖励及时平仓
            elif abs(z_score) > 2.5:  # 超过止损线
                reward *= 0.3  # 显著惩罚过度持仓
            
        # 交易成本惩罚
        if position_changed:
            reward -= 1.0
        
        return np.clip(reward, -10, 10)

    def get_dates(self):
        """获取当前配对的交易日期"""
        stock1 = self.pairs_data.iloc[self.current_pair_idx]['stock1']
        stock1_data = self.data[self.data['stock_code'] == stock1]
        return stock1_data['date'].values
