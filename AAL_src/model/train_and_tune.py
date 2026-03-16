import sys
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb # type: ignore

# 导入滚动窗口划分逻辑
from rolling_window_train import get_window_splits, total_windows

# 模型保存目录
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. 定义模型和超参数搜索空间
models = {
    'RandomForest': {
        'estimator': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
        },
    },
    'LightGBM': {
        'estimator': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [15, 31],
        },
    },
}

# 2. 滚动窗口：标准化 + 调参 + 训练 + 保存
scaler = StandardScaler()
all_results = {name: [] for name in models}
last_models = {}  # 保存最后一个窗口的模型和scaler
print(f"滚动窗口总数：{total_windows}个（步长1天，测试窗口1天）")
print(f"模型：{', '.join(models.keys())}\n")

for w in get_window_splits():
    i = w['window_id']
    test_date = w['test_date']
    X_train, y_train = w['X_train'], w['y_train']
    X_val, y_val = w['X_val'], w['y_val']
    X_test, y_test = w['X_test'], w['y_test']

    # 特征标准化（仅在训练集上 fit）
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    for model_name, cfg in models.items():
        # 超参数调优（GridSearchCV 在训练集内交叉验证）
        grid_search = GridSearchCV(
            cfg['estimator'],
            cfg['param_grid'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        # 保存每个窗口的模型
        model_path = os.path.join(SAVE_DIR, f'{model_name}_window_{i+1}.joblib')
        joblib.dump(best_model, model_path)

        # 记录预测结果（供评估模块使用）
        val_preds = best_model.predict(X_val_scaled)
        test_pred = best_model.predict(X_test_scaled)[0]

        all_results[model_name].append({
            'window': i + 1,
            'test_date': test_date.strftime('%Y-%m-%d'),
            'train_size': len(X_train),
            'best_params': grid_search.best_params_,
            'val_preds': val_preds.tolist(),
            'val_actual': y_val.values.tolist(),
            'test_pred': int(test_pred),
            'test_actual': int(y_test.values[0]),
        })

        # 记录最后一个窗口的模型（用于预测未来）
        last_models[model_name] = best_model

    if i % 50 == 0:
        print(f"窗口 {i+1}/{total_windows}: 测试日 {test_date.strftime('%Y-%m-%d')} 训练完成")

# 3. 保存最后一个窗口的scaler和模型（用于预测2018-02-08）
joblib.dump(scaler, os.path.join(SAVE_DIR, 'last_scaler.joblib'))
for model_name, model in last_models.items():
    joblib.dump(model, os.path.join(SAVE_DIR, f'{model_name}_final.joblib'))

print(f"\n训练完成，共 {total_windows} 个窗口")
print(f"所有模型已保存至：{SAVE_DIR}")

# 4. 保存评估结果到 JSON（供 evaluate.py 直接读取，无需重新训练）
results_path = os.path.join(SAVE_DIR, 'all_results.json')
json.dump({'all_results': all_results, 'total_windows': total_windows}, open(results_path, 'w'), ensure_ascii=False)
print(f"评估数据已保存至：{results_path}")
