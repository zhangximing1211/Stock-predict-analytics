import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 读取并处理数据
df = pd.read_csv('/Users/zhangximing/Desktop/Data/AAL_2013-03-11_to_2018-02-07_all_data.csv')
df['date'] = pd.to_datetime(df['date'])  # 转换日期格式
df = df.sort_values('date').reset_index(drop=True)  # 按时间升序排列
df = df.dropna()  # 删除缺失值（如有）

# 2. 定义目标变量（次日涨跌：1=涨，0=跌）
df['next_day_return'] = df['daily_return'].shift(-1)  # 次日收益率
df['target'] = np.where(df['next_day_return'] > 0, 1, 0)  # 1=涨，0=跌
df = df.dropna(subset=['target'])  # 删除最后一行（无次日数据）

# 3. 选择特征（基于前期筛选的核心特征）
feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                'RSI_14', 'MACD', 'BB_width', 'volatility_20', 'daily_return']
X = df[feature_cols]
y = df['target']
dates = df['date']  # 保留日期用于窗口划分