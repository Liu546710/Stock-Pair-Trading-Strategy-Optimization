import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HedgeDataCollector:
    """基础对冲数据收集器：专注于收集股票和指数数据"""
    
    def __init__(self):
        # 修正股票代码格式
        self.stock_pool = {
            '消费': [
                '600519', '000858', '603288',  # 茅台、五粮液、海天味业
                '600887', '000568', '600132'   # 伊利股份、泸州老窖、重庆啤酒
            ],
            '金融': [
                '600036', '601318', '601166',  # 招商、平安、兴业
                '601398', '601288', '601688'   # 工商银行、农业银行、华泰证券
            ],
            '科技': [
                '000063', '002415', '002594',  # 中兴通讯、海康威视、比亚迪
                '002230', '002475', '002008'   # 科大讯飞、立讯精密、大族激光
            ],
            '医药': [
                '600276', '300015', '000538',  # 恒瑞医药、爱尔眼科、云南白药
                '300760', '603259', '600196'   # 迈瑞医疗、药明康德、复星医药
            ],
            '工业': [
                '600028', '601899', '600019',  # 中国石化、紫金矿业、宝钢股份
                '601857', '600309', '601225'   # 中国石油、万华化学、陕西煤业
            ],
            '地产': [
                '600048', '001979', '600606',  # 保利发展、招商蛇口、绿地控股
                '600340', '000002', '600383'   # 华夏幸福、万科A、金地集团
            ],
            '基建': [
                '601668', '601390', '601800',  # 中国建筑、中国中铁、中国交建
                '600820', '600039', '600012'   # 隧道股份、三一重工、皖通高速
            ],
            '新能源': [
                '601012', '601615', '002460',  # 隆基绿能、明阳智能、赣锋锂业
                '300274', '002129', '300750'   # 阳光电源、中环股份、宁德时代
            ]
        }
        
        # 修改波动指数获取方式
        self.index_pool = {
            '主要指数': ['000300', '000905', '000016'],  # 沪深300、中证500、上证50
            '行业指数': {
                '消费': ['000932'],  # 中证主要消费
                '金融': ['000914'],  # 中证金融
                '科技': ['000993'],  # 中证信息技术
                '医药': ['000991'],  # 中证医药
                '周期': ['000934']   # 中证工业
            },
            '风格指数': ['000922', '000925'],  # 中证红利、中证小盘
            '波动指数': ['000825'],            # 中证全指波动指数
        }
        
        # 时间范围
        self.start_date = '20180101'
        self.end_date = '20240101'
    
    def process_stock_data(self, data):
        """处理股票数据：标准化列名"""
        rename_dict = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        }
        data = data.rename(columns=rename_dict)
        data['date'] = pd.to_datetime(data['date'])
        return data
    
    def process_index_data(self, data):
        """处理指数数据：标准化列名"""
        rename_dict = {
            '日期': 'date',
            '开盘': 'index_open',
            '收盘': 'index_close',
            '最高': 'index_high',
            '最低': 'index_low',
            '成交量': 'index_volume',
            '成交额': 'index_amount'
        }
        data = data.rename(columns=rename_dict)
        data['date'] = pd.to_datetime(data['date'])
        return data
    
    def collect_data(self):
        """收集所有股票和指数数据"""
        all_stock_data = []
        
        # 1. 收集股票数据
        for industry, stocks in self.stock_pool.items():
            for stock_code in stocks:
                try:
                    print(f"获取股票 {stock_code} 数据...")
                    # 确保股票代码格式正确
                    if len(stock_code) != 6:
                        stock_code = stock_code.zfill(6)
                        
                    stock_data = ak.stock_zh_a_hist(
                        symbol=stock_code,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        adjust="qfq"
                    )
                    stock_data = self.process_stock_data(stock_data)
                    stock_data['stock_code'] = stock_code  # 使用补零后的股票代码
                    stock_data['industry'] = industry
                    all_stock_data.append(stock_data)
                    print(f"成功获取股票 {stock_code} 数据")
                except Exception as e:
                    print(f"获取股票 {stock_code} 数据失败: {str(e)}")
        
        # 2. 收集指数数据
        index_data = self.collect_index_data()
        
        # 3. 合并所有股票数据
        combined_stock_data = pd.concat(all_stock_data, ignore_index=True)
        
        # 4. 合并股票和指数数据
        final_data = pd.merge(combined_stock_data, index_data, on='date', how='inner')
        
        # 5. 保存数据
        final_data.to_csv('raw_hedge_data.csv', index=False)
        print("\n数据已保存到 raw_hedge_data.csv")
        print(f"总记录数: {len(final_data)}")
        print(f"股票数量: {final_data['stock_code'].nunique()}")
        print(f"日期范围: {final_data['date'].min()} 到 {final_data['date'].max()}")
        
        return final_data

    def collect_index_data(self):
        """收集所有指数数据"""
        all_index_data = []
        
        # 收集主要指数
        for index_code in self.index_pool['主要指数']:
            try:
                print(f"获取主要指数 {index_code} 数据...")
                index_data = ak.index_zh_a_hist(symbol=index_code)
                index_data = self.process_index_data(index_data)
                index_data['index_type'] = 'main'
                index_data['index_code'] = index_code
                all_index_data.append(index_data)
            except Exception as e:
                print(f"获取指数 {index_code} 数据失败: {str(e)}")
        
        # 收集行业指数
        for industry, codes in self.index_pool['行业指数'].items():
            for index_code in codes:
                try:
                    print(f"获取行业指数 {index_code} ({industry}) 数据...")
                    index_data = ak.index_zh_a_hist(symbol=index_code)
                    index_data = self.process_index_data(index_data)
                    index_data['index_type'] = 'industry'
                    index_data['industry'] = industry
                    index_data['index_code'] = index_code
                    all_index_data.append(index_data)
                except Exception as e:
                    print(f"获取指数 {index_code} 数据失败: {str(e)}")
        
        # 收集风格指数和波动指数
        for index_code in self.index_pool['风格指数'] + self.index_pool['波动指数']:
            try:
                print(f"获取特殊指数 {index_code} 数据...")
                index_data = ak.index_zh_a_hist(symbol=index_code)
                index_data = self.process_index_data(index_data)
                index_data['index_type'] = 'style' if index_code in self.index_pool['风格指数'] else 'volatility'
                index_data['index_code'] = index_code
                all_index_data.append(index_data)
            except Exception as e:
                print(f"获取指数 {index_code} 数据失败: {str(e)}")
        
        return pd.concat(all_index_data, ignore_index=True)

if __name__ == "__main__":
    collector = HedgeDataCollector()
    raw_data = collector.collect_data()


