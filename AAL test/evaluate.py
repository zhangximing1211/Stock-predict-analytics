import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 导入滚动窗口划分（仅数据划分，不触发训练）
_model_dir = os.path.join(os.path.dirname(__file__), '..', 'AAL src', 'model')
if _model_dir not in sys.path:
    sys.path.append(_model_dir)
from rolling_window_train import get_window_splits, total_windows

SAVE_DIR = os.path.join(_model_dir, 'saved_models')
MODEL_NAMES = ['RandomForest', 'LightGBM']

# 输出文件路径
REPORT_PATH = os.path.join(os.path.dirname(__file__), 'evaluation_report.txt')
_report_file = open(REPORT_PATH, 'w', encoding='utf-8')

def log(msg=''):
    """同时输出到终端和文件"""
    print(msg)
    _report_file.write(msg + '\n')

# 1. 逐窗口加载模型 + 推理，收集评估数据
log(f"正在加载 {total_windows} 个窗口的模型并推理...")
all_results = {name: [] for name in MODEL_NAMES}

for w in get_window_splits():
    i = w['window_id']
    test_date = w['test_date']
    X_train, y_train = w['X_train'], w['y_train']
    X_val, y_val = w['X_val'], w['y_val']
    X_test, y_test = w['X_test'], w['y_test']

    # 重新 fit scaler（与训练时一致，StandardScaler 是确定性的）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    for model_name in MODEL_NAMES:
        model_path = os.path.join(SAVE_DIR, f'{model_name}_window_{i+1}.joblib')
        model = joblib.load(model_path)

        val_preds = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)[0]

        all_results[model_name].append({
            'window': i + 1,
            'test_date': test_date.strftime('%Y-%m-%d'),
            'train_size': len(X_train),
            'val_preds': val_preds.tolist(),
            'val_actual': y_val.values.tolist(),
            'test_pred': int(test_pred),
            'test_actual': int(y_test.values[0]),
        })

    if i % 50 == 0:
        log(f"  窗口 {i+1}/{total_windows} 推理完成")

log("推理完成，开始评估...\n")

# 2. 评估报告
log("=" * 60)
log("模型评估报告")
log("=" * 60)

for model_name, results in all_results.items():
    results_df = pd.DataFrame(results)

    y_true = results_df['test_actual'].values
    y_pred = results_df['test_pred'].values

    test_acc = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred, zero_division=0)
    test_recall = recall_score(y_true, y_pred, zero_division=0)
    test_f1 = f1_score(y_true, y_pred, zero_division=0)

    val_accs = [accuracy_score(row['val_actual'], row['val_preds']) for _, row in results_df.iterrows()]
    avg_val_acc = np.mean(val_accs)

    cm = confusion_matrix(y_true, y_pred)

    log(f"\n【{model_name}】")
    log(f"  滚动窗口数：{total_windows}")
    log(f"\n  --- 验证集 ---")
    log(f"  平均验证准确率：{avg_val_acc:.4f}")
    log(f"\n  --- 测试集（逐日预测） ---")
    log(f"  准确率 (Accuracy)：{test_acc:.4f}（{int(test_acc * total_windows)}/{total_windows}）")
    log(f"  精确率 (Precision)：{test_precision:.4f}")
    log(f"  召回率 (Recall)：{test_recall:.4f}")
    log(f"  F1 分数：{test_f1:.4f}")
    log(f"\n  混淆矩阵（行=实际，列=预测）：")
    log(f"          预测跌  预测涨")
    log(f"  实际跌   {cm[0][0]:>4d}   {cm[0][1]:>4d}")
    log(f"  实际涨   {cm[1][0]:>4d}   {cm[1][1]:>4d}")
    log(f"\n  分类报告：")
    log(classification_report(y_true, y_pred, target_names=['跌', '涨']))
    log("-" * 60)

# 3. 模型对比
log("\n模型对比汇总：")
log(f"{'模型':<15} {'验证准确率':<12} {'测试准确率':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
for model_name, results in all_results.items():
    results_df = pd.DataFrame(results)
    y_true = results_df['test_actual'].values
    y_pred = results_df['test_pred'].values
    val_accs = [accuracy_score(row['val_actual'], row['val_preds']) for _, row in results_df.iterrows()]
    log(f"{model_name:<15} "
        f"{np.mean(val_accs):<12.4f} "
        f"{accuracy_score(y_true, y_pred):<12.4f} "
        f"{precision_score(y_true, y_pred, zero_division=0):<12.4f} "
        f"{recall_score(y_true, y_pred, zero_division=0):<12.4f} "
        f"{f1_score(y_true, y_pred, zero_division=0):<12.4f}")

_report_file.close()
print(f"\n评估报告已保存至：{REPORT_PATH}")
