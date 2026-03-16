"""
使用最后一个滚动窗口训练的模型，预测 2018-02-08 的涨跌
逻辑：用 2018-02-07（数据集最后一天）的特征 → 预测次日涨跌
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np

# 路径设置
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'AAL src', 'model', 'saved_models')
sys.path.append(os.path.join(PROJECT_ROOT, 'AAL src', 'model'))

# 1. 加载原始数据（包含 2018-02-07，不做 target dropna）
raw_df = pd.read_csv('/Users/zhangximing/Desktop/Data/AAL_2013-03-11_to_2018-02-07_all_data.csv')
raw_df['date'] = pd.to_datetime(raw_df['date'])
raw_df = raw_df.sort_values('date').reset_index(drop=True)

# 2. 取 2018-02-07 的特征
feature_cols = ['open', 'high', 'low', 'close', 'volume',
                'RSI_14', 'MACD', 'BB_width', 'volatility_20', 'daily_return']

last_day = raw_df[raw_df['date'] == raw_df['date'].max()]
predict_date = last_day['date'].values[0]
X_predict = last_day[feature_cols]

print("=" * 60)
print(f"预测目标日期：2018-02-08（基于 {pd.Timestamp(predict_date).strftime('%Y-%m-%d')} 的特征）")
print("=" * 60)

print(f"\n输入特征：")
for col in feature_cols:
    print(f"  {col}: {X_predict[col].values[0]:.4f}")

# 3. 加载最后一个窗口的 scaler 和模型
scaler = joblib.load(os.path.join(MODEL_DIR, 'last_scaler.joblib'))
X_predict_scaled = scaler.transform(X_predict)

model_names = ['RandomForest', 'LightGBM']
prediction_rows = []
print(f"\n预测结果：")
print("-" * 40)

for model_name in model_names:
    model_path = os.path.join(MODEL_DIR, f'{model_name}_final.joblib')
    model = joblib.load(model_path)

    pred = model.predict(X_predict_scaled)[0]
    pred_proba = model.predict_proba(X_predict_scaled)[0]

    direction = '涨 ↑' if pred == 1 else '跌 ↓'
    confidence = pred_proba[int(pred)]

    prediction_rows.append({
        'model': model_name,
        'predict_date': '2018-02-08',
        'based_on_date': pd.Timestamp(predict_date).strftime('%Y-%m-%d'),
        'prediction': int(pred),
        'direction': '涨' if pred == 1 else '跌',
        'prob_down': round(pred_proba[0], 4),
        'prob_up': round(pred_proba[1], 4),
        'confidence': round(confidence, 4),
    })

    print(f"\n  【{model_name}】")
    print(f"    预测方向：{direction}")
    print(f"    预测概率：跌 {pred_proba[0]:.4f} | 涨 {pred_proba[1]:.4f}")
    print(f"    置信度：{confidence:.4f}")

# 4. 保存预测结果到 CSV
results_df = pd.DataFrame(prediction_rows)
output_path = os.path.join(PROJECT_ROOT, 'AAL test', 'prediction_results.csv')
results_df.to_csv(output_path, index=False)
print(f"\n预测结果已保存至：{output_path}")

print("\n" + "=" * 60)
print("注意：此预测基于历史规律，不构成投资建议")
print("=" * 60)
