import sys
import os
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

# 加载预处理模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.preprocess_data import df, feature_cols, X, y, dates

# 1. 窗口参数
initial_train_months = 36  # 初始训练窗口：36个月
val_months_len = 1         # 验证窗口：1个月
# 滚动步长：1天，测试窗口：1天

# 2. 获取所有交易日（已按日期升序）
all_dates = df['date'].values

# 3. 确定可用的测试日：需要测试日前有 36个月训练 + 1个月验证 的数据
min_date = df['date'].min()
first_test_date = min_date + relativedelta(months=initial_train_months + val_months_len)

# 筛选出所有可用的测试日
test_dates = df.loc[df['date'] >= first_test_date, 'date'].unique()
total_windows = len(test_dates)


def get_window_splits():
    """生成滚动窗口划分，返回每个窗口的训练/验证/测试数据。"""
    for i, test_date in enumerate(test_dates):
        test_date = pd.Timestamp(test_date)

        # 验证期：测试日前1个月
        val_end = test_date - pd.Timedelta(days=1)
        val_start = test_date - relativedelta(months=val_months_len)

        # 训练期：验证期前36个月
        train_end = val_start - pd.Timedelta(days=1)
        train_start = val_start - relativedelta(months=initial_train_months)

        # 按日期筛选数据
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        val_mask = (df['date'] >= val_start) & (df['date'] <= val_end)
        test_mask = df['date'] == test_date

        yield {
            'window_id': i,
            'test_date': test_date,
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
            'X_train': X[train_mask], 'y_train': y[train_mask],
            'X_val': X[val_mask], 'y_val': y[val_mask],
            'X_test': X[test_mask], 'y_test': y[test_mask],
        }


# 直接运行时打印窗口划分信息
if __name__ == '__main__':
    print(f"滚动窗口总数：{total_windows}个（步长1天，测试窗口1天）")
    for w in get_window_splits():
        i = w['window_id']
        if i % 50 == 0:
            print(f"窗口 {i+1}/{total_windows}: "
                  f"训练 {w['train_start'].strftime('%Y-%m-%d')}~{w['train_end'].strftime('%Y-%m-%d')} ({len(w['X_train'])}条) | "
                  f"验证 {w['val_start'].strftime('%Y-%m-%d')}~{w['val_end'].strftime('%Y-%m-%d')} ({len(w['X_val'])}条) | "
                  f"测试 {w['test_date'].strftime('%Y-%m-%d')} ({len(w['X_test'])}条)")
    print(f"\n滚动窗口划分完成，共 {total_windows} 个窗口")
