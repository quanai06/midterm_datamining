"""Hàm dùng chung cho các câu b-e (chuyển nguyên văn từ 02_data_understanding.py, không đổi hành vi).

Mọi định nghĩa "cách biểu diễn khoảng cách", luật miền, đẳng thức nghiệp vụ, Hopkins, kNN đều nằm ở đây để câu b và câu c dùng đúng
cùng một phép đo. Module này không đọc file dữ liệu nào.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

# Luật miền nghiệp vụ (giả định của người phân tích, không lấy từ bộ sinh): (min, max, min_có_đóng)
RULES = {"age": (18, 100, True), "n_orders_12m": (1, 500, True), "avg_order_value": (0, 10000, False),
         "total_spend_12m": (0, 1e6, False), "recency_days": (0, 365, True), "discount_usage_rate": (0, 1, True),
         "sessions_per_month": (0, 100, True), "avg_session_minutes": (0, 240, False),
         "pages_per_session": (0, 100, True), "cart_abandon_rate": (0, 1, True)}


def viol_mask(df, num):
    out = pd.DataFrame(False, index=df.index, columns=num)
    for c in num:
        lo, hi, closed = RULES[c]
        x = df[c].astype(float)
        out[c] = ((x < lo) if closed else (x <= lo)) | (x > hi)
    return out


def identity_dev(df):
    return np.log(df["total_spend_12m"].astype(float) / (df["n_orders_12m"].astype(float) * df["avg_order_value"].astype(float)))


def identity_flag(df, k=5):
    """Cờ vi phạm đẳng thức xấp xỉ total ~ n x AOV: |ln(total/(n x AOV)) - trung vị| > k x sd bền (MAD x 1,4826)."""
    dev = identity_dev(df)
    med, sd = dev.median(), 1.4826 * (dev - dev.median()).abs().median()
    return (dev - med).abs() > k * sd, dev, med, sd


def select_log_cols(df, dd, num):
    """Cột cần log: skew sau khi loại ô vi phạm luật miền > 1 và không phải biến tỉ lệ bị chặn [0,1]."""
    X = df[num].astype(float)
    sk = pd.DataFrame({"skew_excl_violations": X.where(~viol_mask(df, num)).skew(),
                       "bounded_[0,1]": [dd.set_index("attribute").loc[c, "valid_domain"] == "[0,1]" for c in num]}, index=num)
    return [c for c in num if sk.loc[c, "skew_excl_violations"] > 1 and not sk.loc[c, "bounded_[0,1]"]], sk


def bowley(x):
    q1, med, q3 = x.quantile(.25), x.median(), x.quantile(.75)
    return (q3 + q1 - 2 * med) / (q3 - q1) if q3 > q1 else 0.0


def make_lens(df, num, kind, log_cols):
    """Ma trận để TÍNH khoảng cách (chỉ dùng cho phân tích, không phải dữ liệu tiền xử lý): điền trung vị tạm thời."""
    X = df[num].astype(float).copy()
    X = X.fillna(X.median())
    if kind == "raw_z":
        return ((X - X.mean()) / X.std(ddof=0)).to_numpy()
    for c in log_cols:
        X[c] = np.log1p(X[c].clip(lower=0))
    iqr = (X.quantile(.75) - X.quantile(.25)).replace(0, 1.0)
    return ((X - X.median()) / iqr).to_numpy()


def hopkins(X, seed, reps=30):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    m = max(10, n // 10)
    tree = cKDTree(X)
    lo, hi = X.min(0), X.max(0)
    hs = []
    for _ in range(reps):
        idx = rng.choice(n, m, replace=False)
        w = tree.query(X[idx], k=2)[0][:, 1]
        u = tree.query(rng.uniform(lo, hi, (m, d)), k=1)[0]
        hs.append(u.sum() / (u.sum() + w.sum()))
    return float(np.mean(hs)), float(np.std(hs))


def knn_dist(X, k=5):
    return cKDTree(X).query(X, k=k + 1)[0][:, k]


def knn_sets(X, k=5):
    idx = cKDTree(X).query(X, k=k + 1)[1][:, 1:]
    return [set(r) for r in idx]


def pair_share(X, rows):
    D2 = squareform(pdist(X)) ** 2
    m = np.zeros(len(X), bool)
    m[list(rows)] = True
    return float((D2 * (m[:, None] | m[None, :])).sum() / D2.sum())


def pca2(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    return Xc @ Vt[:2].T, var
