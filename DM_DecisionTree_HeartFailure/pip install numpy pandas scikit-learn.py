#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于决策树的心力衰竭医学数据分类实验（带 result.csv 结果保存）
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def load_data(file_path: str):
    """
    从 CSV 文件加载数据。
    参数：
        file_path: str, 数据集路径
    返回：
        X: 特征矩阵 (numpy.ndarray)
        y: 标签向量 (numpy.ndarray)
    """
    data = pd.read_csv(file_path)
    # 默认：最后一列为标签 DEATH_EVENT
    X = data.iloc[:, :-1].values  # 特征
    y = data.iloc[:, -1].values   # 标签
    return X, y


def evaluate_model(
    X,
    y,
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
    max_depth=None,
    min_samples_leaf: int = 1,
):
    """
    使用分层十折交叉验证评估决策树模型。

    返回：
        avg_acc, avg_precision, avg_recall, avg_f1
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    metrics = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 决策树模型，关键参数写成变量方便记录
        model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion="gini"  # 默认基尼系数，可改为 "entropy"
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        metrics.append((acc, precision, recall, f1))

    avg_metrics = np.mean(metrics, axis=0)
    return avg_metrics  # acc, precision, recall, f1


def save_result_to_csv(
    result_path: str,
    params: dict,
    metrics: dict,
):
    """
    将参数 + 结果保存到 result.csv 中。
    如果文件不存在，写入表头；存在则追加一行。
    """
    path = Path(result_path)

    # 合并参数与指标为一行
    row = {**params, **metrics}
    df = pd.DataFrame([row])

    # mode='a' 追加写入；header=not path.exists() 控制是否写表头
    df.to_csv(
        path,
        index=False,
        mode='a',
        header=not path.exists()
    )


if __name__ == "__main__":
    # 数据文件（脚本和 csv 在同一目录时写文件名即可）
    data_path = "heart_failure_clinical_records_dataset.csv"

    # ========= 在这里设定本次实验使用的参数 =========
    n_splits = 10
    shuffle = True
    random_state = 42
    max_depth = None          # 例如可以改成 4、5 等
    min_samples_leaf = 1      # 例如可以改成 3、5 等
    criterion = "gini"        # 决策树划分标准
    # =============================================

    print("Loading dataset...")
    X, y = load_data(data_path)

    print("Evaluating Decision Tree model with 10-fold CV...")
    avg_acc, avg_precision, avg_recall, avg_f1 = evaluate_model(
        X,
        y,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    print("Average Metrics over 10 folds:")
    print(f"Accuracy : {avg_acc:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall   : {avg_recall:.4f}")
    print(f"F1-Score : {avg_f1:.4f}")

    # ===== 把参数 + 结果写入 result.csv =====
    params_dict = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
        "n_splits": n_splits,
        "shuffle": shuffle,
    }

    metrics_dict = {
        "accuracy": avg_acc,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1,
    }

    save_result_to_csv("result.csv", params_dict, metrics_dict)
    print('Results have been saved to "result.csv".')