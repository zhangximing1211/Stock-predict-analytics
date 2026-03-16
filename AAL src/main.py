"""
AAL 股票涨跌预测 - 主执行入口
=====================================
执行流程：
  1. 数据预处理（preprocess_data.py）
  2. 滚动窗口划分（rolling_window_train.py）
  3. 模型训练与调参 + 保存（train_and_tune.py）
  4. 模型评估（evaluate.py）
  5. 预测 2018-02-08 涨跌（predict_next_day.py）
"""

import sys
import os
import time

# 设置模块搜索路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'AAL src', 'model')
TEST_DIR = os.path.join(PROJECT_ROOT, 'AAL test')

sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, TEST_DIR)

if __name__ == '__main__':
    print("=" * 60)
    print("AAL 股票涨跌预测系统")
    print("=" * 60)

    # Step 1: 数据预处理（导入时自动执行）
    print("\n[Step 1] 数据预处理...")
    from data.preprocess_data import df, X, y, feature_cols
    print(f"  数据量：{len(df)} 条")
    print(f"  特征数：{len(feature_cols)} 个 → {feature_cols}")
    print(f"  日期范围：{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  目标分布：涨 {int(y.sum())}, 跌 {int(len(y) - y.sum())}")

    # Step 2: 滚动窗口划分
    print("\n[Step 2] 滚动窗口划分...")
    from rolling_window_train import total_windows
    print(f"  滚动窗口总数：{total_windows} 个（步长1天，测试窗口1天）")

    # Step 3: 模型训练与调参 + 保存
    print("\n[Step 3] 模型训练与超参数调优...")
    start_time = time.time()
    from train_and_tune import all_results, total_windows
    elapsed = time.time() - start_time
    print(f"  训练耗时：{elapsed:.1f} 秒")

    # Step 4: 模型评估
    print("\n[Step 4] 模型评估...")
    import evaluate

    # Step 5: 预测 2018-02-08
    print("\n[Step 5] 预测未来...")
    import predict_next_day

    print("\n" + "=" * 60)
    print("全部流程执行完毕")
    print("=" * 60)
