#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def load_data(file_path):
    """
    从 CSV 文件加载数据，默认最后一列是标签 DEATH_EVENT。
    """
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # 所有特征列
    y = data.iloc[:, -1].values   # 最后一列标签
    return X, y


def evaluate_decision_tree(X, y,
                           n_splits=10,
                           shuffle=True,
                           random_state=42,
                           max_depth=None,
                           min_samples_leaf=1,
                           criterion="gini"):
    """
    用分层 K 折交叉验证评估决策树，返回四个平均指标。
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    metrics = []  # 每折保存 (acc, prec, rec, f1)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )

        metrics.append((acc, precision, recall, f1))

    metrics = np.array(metrics)
    avg_acc, avg_prec, avg_rec, avg_f1 = metrics.mean(axis=0)
    return avg_acc, avg_prec, avg_rec, avg_f1


def save_result_csv(result_path,
                    params,
                    metrics):
    """
    将参数 + 结果追加写入 result.csv。
    若文件不存在则写入表头，存在则仅追加新行。
    """
    row = {}
    row.update(params)
    row.update(metrics)

    df = pd.DataFrame([row])

    file_exists = os.path.exists(result_path)

    df.to_csv(
        result_path,
        index=False,
        mode="a",
        header=not file_exists
    )


if __name__ == "__main__":
    # ========== 1. 数据路径 ==========
    # 确保这个 csv 文件和本脚本在同一目录下
    data_path = "heart_failure_clinical_records_dataset.csv"

    # ========== 2. 决策树参数（这里可以自己改） ==========
    n_splits = 10          # 交叉验证折数
    shuffle = True         # 是否打乱
    random_state = 42      # 随机种子
    max_depth = None       # 限制树深度，如 4、5 等；None 表示不限制
    min_samples_leaf = 1   # 叶子节点最小样本数
    criterion = "gini"     # "gini" 或 "entropy"
    # ====================================

    print("Loading dataset...")
    X, y = load_data(data_path)

    print("Evaluating Decision Tree with 10-fold CV...")
    avg_acc, avg_prec, avg_rec, avg_f1 = evaluate_decision_tree(
        X, y,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )

    print("Average Metrics over {} folds:".format(n_splits))
    print("Accuracy : {:.4f}".format(avg_acc))
    print("Precision: {:.4f}".format(avg_prec))
    print("Recall   : {:.4f}".format(avg_rec))
    print("F1-Score : {:.4f}".format(avg_f1))

    # ========== 3. 保存结果到 result.csv ==========
    params_dict = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
        "n_splits": n_splits,
        "shuffle": shuffle
    }

    metrics_dict = {
        "accuracy": avg_acc,
        "precision": avg_prec,
        "recall": avg_rec,
        "f1_score": avg_f1
    }

    save_result_csv("result.csv", params_dict, metrics_dict)
    print('Results have been saved to "result.csv".')