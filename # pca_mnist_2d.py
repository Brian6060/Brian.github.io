#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---- minimal bootstrap: make sure deps exist ----
import sys, importlib, subprocess
def ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])
        importlib.invalidate_caches()
for pkg, imp in [("numpy", None), ("scikit-learn", "sklearn"),
                 ("matplotlib", None), ("liac-arff", "arff")]:
    ensure(pkg, imp)

# ---- imports ----
import numpy as np
import matplotlib
matplotlib.use("Agg")               # 无需 GUI，直接保存图片
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_openml

# ---- PCA via SVD (与教程一致) ----
def pca_fit_svd(X):                 # X: (features m) x (samples n)
    m, n = X.shape
    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean
    Y = Xc.T / np.sqrt(max(n - 1, 1))      # Y = X^T / sqrt(n-1)
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    PCs = Vt.T                              # 列是主成分方向
    return PCs, mean, Xc

def pca_project(PCs, Xc, k):
    return PCs[:, :k].T @ Xc               # Z in R^{k x n}

# ---- 绘图 ----
MARKERS = ['o','s','^','v','<','>','P','X','D','*']
def plot_2d(Ztr2, ytr, Zte2, yte, note, out):
    plt.figure(figsize=(8,7))
    for i, c in enumerate(sorted(np.unique(ytr))):
        idx = (ytr == c)
        plt.scatter(Ztr2[0, idx], Ztr2[1, idx], s=10, alpha=0.7,
                    marker=MARKERS[i % len(MARKERS)], label=f"train {c}")
    for i, c in enumerate(sorted(np.unique(yte))):
        idx = (yte == c)
        plt.scatter(Zte2[0, idx], Zte2[1, idx], s=16, alpha=0.9,
                    marker=MARKERS[i % len(MARKERS)], facecolors='none', edgecolors='k',
                    label=f"test {c}")
    plt.title("PCA → 2D (k=2)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    plt.text(0.99, 0.01, note, transform=plt.gca().transAxes,
             ha='right', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[ok] saved: {out}")

# ---- 数据加载 ----
def load_mnist_2000():
    """用 OpenML 取 MNIST（不依赖 pandas）"""
    try:
        ds = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    except TypeError:
        # 旧版 sklearn 没有 parser 参数，安装 pandas 再取
        ensure("pandas")
        ds = fetch_openml("mnist_784", version=1, as_frame=False)
    X = ds.data.astype(np.float64) / 255.0
    y = ds.target
    y = y.astype(int) if hasattr(y, "astype") else np.asarray(y, dtype=int)
    Xtr, ytr = X[:60000][:2000], y[:60000][:2000]
    Xte, yte = X[60000:][:2000], y[60000:][:2000]
    return Xtr, ytr, Xte, yte

def load_digits_fallback():
    """离线兜底：sklearn digits (8×8)"""
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(int)
    n_tr = min(1200, X.shape[0] - 300)
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xte, yte = X[n_tr:n_tr+600], y[n_tr:n_tr+600]
    return Xtr, ytr, Xte, yte

# ---- 主流程 ----
def main():
    out = Path("mnist_pca_2d.png")
    # 1) 取数据
    try:
        print("[info] fetching MNIST from OpenML …")
        Xtr, ytr, Xte, yte = load_mnist_2000()
        note = "Dataset: MNIST (28×28)"
    except Exception as e:
        print("[warn] MNIST fetch failed, using digits fallback.", repr(e))
        Xtr, ytr, Xte, yte = load_digits_fallback()
        note = "Dataset: sklearn digits (8×8) — fallback"
    # 2) 形状：行=特征，列=样本
    Xtr_, Xte_ = Xtr.T, Xte.T
    # 3) 训练集拟合 PCA（SVD）
    PCs, mean_tr, Xtrc = pca_fit_svd(Xtr_)
    # 4) 投影训练/测试到前两主成分
    Ztr2 = pca_project(PCs, Xtrc, k=2)
    Zte2 = pca_project(PCs, Xte_.copy() - mean_tr, k=2)
    # 5) 画图保存
    plot_2d(Ztr2, ytr, Zte2, yte, note, str(out))

if __name__ == "__main__":
    main()