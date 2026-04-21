import pandas as pd
import numpy as np
import traceback
from pathlib import Path
import os
from statsmodels.tsa.stattools import coint
from scipy import stats
import random

class DataPreprocessor:
    def __init__(self, raw_data_path, output_path):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.processed_data = None
        self.pairs_data = None
        
    def process(self):
        """主处理流程"""
        print("\n1. 开始基础数据清理...")
        self._load_and_clean_data()
        
        print("\n2. 开始计算对冲指标...")
        self._calculate_hedge_indicators()
        
        print("\n3. 开始筛选对冲配对...")
        self._select_hedge_pairs()
        
        print("\n4. 开始计算市场环境指标...")
        self._calculate_market_indicators()
        
        # 保存处理后的数据
        self.processed_data.to_csv(self.output_path, index=False)
        print(f"\n数据处理完成，已保存到: {self.output_path}")

    def _load_and_clean_data(self):
        """加载并清理数据"""
        self.processed_data = pd.read_csv(self.raw_data_path)
        
        # 数据类型转换
        self.processed_data['date'] = pd.to_datetime(self.processed_data['date'])
        
        # 确保数值列为数值类型
        numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 'pct_change']
        for col in numeric_columns:
            self.processed_data[col] = pd.to_numeric(self.processed_data[col], errors='coerce')
        
        # 按日期和股票代码排序
        self.processed_data.sort_values(['date', 'stock_code'], inplace=True)
        
        # 删除缺失值 - 只删除关键列的缺失值，而不是整行
        self.processed_data.dropna(subset=['close', 'pct_change'], inplace=True)
        
        # 记录原始数据量
        original_count = len(self.processed_data)
        print(f"原始数据行数: {original_count}")
        
        # 检查重复数据 - 只检查关键列
        key_columns = ['date', 'stock_code', 'open', 'close', 'high', 'low']
        duplicates = self.processed_data.duplicated(subset=key_columns, keep='first')
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            print(f"发现 {duplicate_count} 行重复数据 (基于关键列)")
            self.processed_data = self.processed_data[~duplicates]
        
        # 确保数据连续性 - 为每个股票创建完整的日期范围
        all_dates = pd.date_range(start=self.processed_data['date'].min(), 
                                 end=self.processed_data['date'].max(), 
                                 freq='B')  # 'B'表示工作日
        
        stocks = self.processed_data['stock_code'].unique()
        
        # 创建完整的日期-股票组合
        date_stock_product = pd.MultiIndex.from_product(
            [all_dates, stocks], 
            names=['date', 'stock_code']
        )
        
        full_index_df = pd.DataFrame(index=date_stock_product).reset_index()
        
        # 合并现有数据
        self.processed_data = pd.merge(
            full_index_df, 
            self.processed_data, 
            on=['date', 'stock_code'], 
            how='left'
        )
        
        # 按股票分组并前向填充缺失值
        self.processed_data = self.processed_data.groupby('stock_code').apply(
            lambda x: x.sort_values('date').ffill()
        ).reset_index(drop=True)
        
        # 再次向后填充可能仍然存在的缺失值
        self.processed_data = self.processed_data.groupby('stock_code').apply(
            lambda x: x.sort_values('date').bfill()
        ).reset_index(drop=True)
        
        # 对于仍然缺失的值，使用合理的默认值填充
        self.processed_data['close'].fillna(method='ffill', inplace=True)
        self.processed_data['open'].fillna(self.processed_data['close'], inplace=True)
        self.processed_data['high'].fillna(self.processed_data['close'], inplace=True)
        self.processed_data['low'].fillna(self.processed_data['close'], inplace=True)
        self.processed_data['volume'].fillna(0, inplace=True)
        self.processed_data['amount'].fillna(0, inplace=True)
        self.processed_data['pct_change'].fillna(0, inplace=True)
        
        # 报告处理后的数据量
        processed_count = len(self.processed_data)
        print(f"处理后数据行数: {processed_count}")
        print(f"数据变化: {processed_count - original_count} 行 ({(processed_count/original_count - 1)*100:.2f}%)")

    def _calculate_hedge_indicators(self):
        """计算对冲相关指标"""
        print("开始计算技术指标...")
        
        # 计算每只股票的技术指标
        for stock_code in self.processed_data['stock_code'].unique():
            print(f"处理股票 {stock_code} 的技术指标...")
            stock_data = self.processed_data[self.processed_data['stock_code'] == stock_code].copy()
            
            if len(stock_data) < 20:
                print(f"警告: 股票 {stock_code} 数据不足，跳过")
                continue
            
            # 1. 计算波动率（多个周期）
            for period in [5, 10, 20]:
                stock_data[f'volatility_{period}d'] = stock_data['pct_change'].rolling(period).std() * np.sqrt(252)  # 年化
            
            # 2. 计算动量（多个周期）
            for period in [5, 10, 20]:
                stock_data[f'momentum_{period}d'] = stock_data['pct_change'].rolling(period).sum()
            
            # 3. 计算RSI（多个周期）
            for period in [6, 14]:
                delta = stock_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                stock_data[f'rsi_{period}d'] = 100 - (100 / (1 + rs))
            
            # 4. 计算移动平均线
            for period in [5, 10, 20, 60]:
                stock_data[f'ma{period}'] = stock_data['close'].rolling(period).mean()
            
            # 5. 计算布林带
            stock_data['ma20'] = stock_data['close'].rolling(20).mean()
            std20 = stock_data['close'].rolling(20).std()
            stock_data['upper_band'] = stock_data['ma20'] + (2 * std20)
            stock_data['lower_band'] = stock_data['ma20'] - (2 * std20)
            
            # 6. 计算MACD
            ema12 = stock_data['close'].ewm(span=12, adjust=False).mean()
            ema26 = stock_data['close'].ewm(span=26, adjust=False).mean()
            stock_data['macd'] = ema12 - ema26
            stock_data['macd_signal'] = stock_data['macd'].ewm(span=9, adjust=False).mean()
            stock_data['macd_hist'] = stock_data['macd'] - stock_data['macd_signal']
            
            # 7. 计算相对强度（相对于行业指数）
            if 'index_close' in stock_data.columns:
                stock_data['rel_strength'] = stock_data['close'].pct_change() - stock_data['index_close'].pct_change()
                stock_data['rel_strength_ma10'] = stock_data['rel_strength'].rolling(10).mean()
            
            # 8. 计算ATR (Average True Range)
            high_low = stock_data['high'] - stock_data['low']
            high_close = np.abs(stock_data['high'] - stock_data['close'].shift())
            low_close = np.abs(stock_data['low'] - stock_data['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            stock_data['atr'] = true_range.rolling(14).mean()
            
            # 9. 计算价格变化率
            for period in [1, 3, 5, 10, 20]:
                stock_data[f'price_change_{period}d'] = stock_data['close'].pct_change(period)
            
            # 10. 计算成交量变化
            stock_data['volume_change'] = stock_data['volume'].pct_change()
            stock_data['volume_ma10'] = stock_data['volume'].rolling(10).mean()
            stock_data['volume_ratio'] = stock_data['volume'] / stock_data['volume_ma10']
            
            # 填充缺失值
            stock_data = stock_data.replace([np.inf, -np.inf], np.nan)
            stock_data = stock_data.fillna(method='ffill')
            stock_data = stock_data.fillna(0)  # 对于仍然缺失的值使用0填充
            
            # 更新数据
            self.processed_data.update(stock_data)
        
        # 删除仍然包含NaN的行
        nan_count = self.processed_data.isna().sum().sum()
        if nan_count > 0:
            print(f"警告: 仍有 {nan_count} 个缺失值，将被填充为0")
            self.processed_data = self.processed_data.fillna(0)
        
        print("技术指标计算完成")

    def _select_hedge_pairs(self):
        """筛选对冲配对"""
        pairs_results = []
        
        # 1. 先进行同行业内配对
        for industry in self.processed_data['industry_x'].unique():
            # 获取同行业的股票
            industry_data = self.processed_data[self.processed_data['industry_x'] == industry]
            stock_codes = industry_data['stock_code'].unique()
            
            print(f"\n处理 {industry} 行业，共有 {len(stock_codes)} 只股票")
            
            # 在同行业内生成配对
            for i in range(len(stock_codes)):
                for j in range(i+1, len(stock_codes)):
                    stock1 = stock_codes[i]
                    stock2 = stock_codes[j]
                    
                    # 获取两只股票的收盘价序列
                    stock1_data = industry_data[industry_data['stock_code'] == stock1]
                    stock2_data = industry_data[industry_data['stock_code'] == stock2]
                    
                    # 确保两只股票的数据长度相同
                    common_dates = sorted(list(set(stock1_data['date']) & set(stock2_data['date'])))
                    if len(common_dates) < 60:  # 至少需要60个交易日的数据
                        continue
                        
                    stock1_prices = stock1_data.set_index('date').loc[common_dates, 'close'].values
                    stock2_prices = stock2_data.set_index('date').loc[common_dates, 'close'].values
                    
                    # 计算相关性
                    correlation = np.corrcoef(stock1_prices, stock2_prices)[0, 1]
                    
                    # 计算协整性
                    try:
                        score, pvalue, _ = coint(stock1_prices, stock2_prices)
                        
                        # 计算价格比率
                        price_ratio = stock1_prices / stock2_prices
                        ratio_mean = np.mean(price_ratio)
                        ratio_std = np.std(price_ratio)
                        
                        # 计算z-score
                        current_ratio = price_ratio[-1]
                        z_score = (current_ratio - ratio_mean) / ratio_std
                        
                        # 放宽筛选条件
                        if (pvalue < 0.15 and  # 协整显著性要求降低 0.1->0.15
                            abs(correlation) > 0.5 and  # 相关性要求降低 
                            abs(z_score) > 0.7):  # z-score要求降低 0.8->0.7
                            
                            pairs_results.append({
                                'industry': industry,
                                'stock1': stock1,
                                'stock2': stock2,
                                'correlation': correlation,
                                'coint_pvalue': pvalue,
                                'price_ratio': current_ratio,
                                'ratio_mean': ratio_mean,
                                'ratio_std': ratio_std,
                                'z_score': z_score
                            })
                    except Exception as e:
                        print(f"处理股票对 {stock1}-{stock2} 时出错: {str(e)}")
                        continue
        
        # 2. 添加跨行业配对
        print("\n开始寻找跨行业配对...")
        all_stocks = self.processed_data['stock_code'].unique()
        
        # 随机选择一部分跨行业组合进行测试
        cross_industry_pairs = []
        for i in range(len(all_stocks)):
            for j in range(i+1, len(all_stocks)):
                stock1 = all_stocks[i]
                stock2 = all_stocks[j]
                
                # 确保是跨行业的
                stock1_industry = self.processed_data[self.processed_data['stock_code'] == stock1]['industry_x'].iloc[0]
                stock2_industry = self.processed_data[self.processed_data['stock_code'] == stock2]['industry_x'].iloc[0]
                
                if stock1_industry != stock2_industry:
                    cross_industry_pairs.append((stock1, stock2))
        
        # 如果跨行业组合太多，随机选择一部分
        if len(cross_industry_pairs) > 200:
            cross_industry_pairs = random.sample(cross_industry_pairs, 200)
        
        # 分析跨行业配对
        for stock1, stock2 in cross_industry_pairs:
            # 获取两只股票的收盘价序列
            stock1_data = self.processed_data[self.processed_data['stock_code'] == stock1]
            stock2_data = self.processed_data[self.processed_data['stock_code'] == stock2]
            
            # 确保两只股票的数据长度相同
            common_dates = sorted(list(set(stock1_data['date']) & set(stock2_data['date'])))
            if len(common_dates) < 60:  # 至少需要60个交易日的数据
                continue
                
            stock1_prices = stock1_data.set_index('date').loc[common_dates, 'close'].values
            stock2_prices = stock2_data.set_index('date').loc[common_dates, 'close'].values
            
            # 计算相关性
            correlation = np.corrcoef(stock1_prices, stock2_prices)[0, 1]
            
            # 计算协整性
            try:
                score, pvalue, _ = coint(stock1_prices, stock2_prices)
                
                # 计算价格比率
                price_ratio = stock1_prices / stock2_prices
                ratio_mean = np.mean(price_ratio)
                ratio_std = np.std(price_ratio)
                
                # 计算z-score
                current_ratio = price_ratio[-1]
                z_score = (current_ratio - ratio_mean) / ratio_std
                
                # 放宽筛选条件
                if (pvalue < 0.15 and  # 更宽松的协整显著性
                    abs(correlation) > 0.4 and  # 更宽松的相关性 0.4->0.6
                    abs(z_score) > 0.7):  # 更宽松的z-score 0.7->0.8
                    
                    pairs_results.append({
                        'industry': f"{stock1_industry}-{stock2_industry}",  # 标记为跨行业
                        'stock1': stock1,
                        'stock2': stock2,
                        'correlation': correlation,
                        'coint_pvalue': pvalue,
                        'price_ratio': current_ratio,
                        'ratio_mean': ratio_mean,
                        'ratio_std': ratio_std,
                        'z_score': z_score
                    })
            except Exception as e:
                print(f"处理股票对 {stock1}-{stock2} 时出错: {str(e)}")
                continue
        
        # 转换为DataFrame并排序
        self.pairs_data = pd.DataFrame(pairs_results)
        if len(self.pairs_data) > 0:
            self.pairs_data = self.pairs_data.sort_values('z_score', ascending=False)
            
            # 保存配对结果
            pairs_path = os.path.join(os.path.dirname(self.output_path), 'hedge_pairs.csv')
            self.pairs_data.to_csv(pairs_path, index=False)
            
            print(f"\n找到 {len(self.pairs_data)} 个潜在的对冲配对")
            print("\n排名前5的对冲机会：")
            print(self.pairs_data.head())
        else:
            print("\n未找到满足条件的对冲配对")

    def _calculate_market_indicators(self):
        """计算市场环境指标"""
        print("开始计算市场环境指标...")
        
        # 获取所有日期
        all_dates = sorted(self.processed_data['date'].unique())
        market_indicators = pd.DataFrame({'date': all_dates})
        
        # 使用沪深300指数数据计算市场指标
        if 'index_close' in self.processed_data.columns:
            # 按日期分组获取指数数据
            index_data = self.processed_data.groupby('date')['index_close'].mean().reset_index()
            market_indicators = pd.merge(market_indicators, index_data, on='date', how='left')
            
            # 计算市场趋势
            market_indicators['market_trend_5d'] = market_indicators['index_close'].pct_change(5)
            market_indicators['market_trend_20d'] = market_indicators['index_close'].pct_change(20)
            
            # 计算市场波动性
            market_indicators['market_volatility'] = market_indicators['index_close'].pct_change().rolling(20).std() * np.sqrt(252)
            
            # 计算市场动量
            market_indicators['market_momentum'] = np.sign(market_indicators['index_close'].diff(20))
            
            # 计算市场RSI
            delta = market_indicators['index_close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            market_indicators['market_rsi'] = 100 - (100 / (1 + rs))
            
            # 计算市场超买超卖指标
            market_indicators['market_overbought'] = (market_indicators['market_rsi'] > 70).astype(int)
            market_indicators['market_oversold'] = (market_indicators['market_rsi'] < 30).astype(int)
        
        # 填充缺失值
        market_indicators = market_indicators.fillna(method='ffill').fillna(0)
        
        # 将市场指标合并到原始数据
        self.processed_data = pd.merge(self.processed_data, market_indicators, on='date', how='left')
        
        print("市场环境指标计算完成")

def main():
    try:
        current_dir = Path(__file__).resolve().parent
        data_dir = current_dir.parent / "data"
        os.makedirs(data_dir, exist_ok=True)
        
        raw_data_path = data_dir / "raw_hedge_data.csv"
        output_path = data_dir / "processed_hedge_data.csv"
        
        if not raw_data_path.exists():
            print(f"错误：找不到原始数据文件: {raw_data_path}")
            return
            
        preprocessor = DataPreprocessor(raw_data_path, output_path)
        preprocessor.process()
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
