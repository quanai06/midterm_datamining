#!/usr/bin/env python3
"""Câu a - Sinh dữ liệu giả lập phân cụm khách hàng thương mại điện tử.

Chuỗi dữ liệu tích lũy (mỗi bước thêm đúng một loại vấn đề):
    data_clean_reference.csv   D0  dữ liệu nền sạch (chỉ để đối chiếu)
    data_after_noise.csv       D1  D0 + nhiễu ngẫu nhiên
    data_after_outliers.csv    D2  D1 + ngoại lai
    data_raw_corrupted.csv     D3  D2 + giá trị thiếu  (đầu vào duy nhất của câu b-e)

Thông tin ẩn (nhãn phân khúc, cờ cầu nối, đặc điểm ẩn, nhật ký chèn lỗi) nằm riêng trong data/hidden/,
KHÔNG có mặt trong bất kỳ file D0-D3 nào.

Chạy:  python src/01_generate_data.py            (local)
       %run 01_generate_data.py                  (Colab; xem notebooks/01_generate_data_colab.ipynb)
Chỉ cần NumPy + Pandas. scikit-learn (nếu có) chỉ dùng cho phần kiểm tra nghiệm thu.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------
# 1. Ràng buộc theo mã sinh viên
# ----------------------------------------------------------------------------------------------
STUDENT_ID = 24022433
SEED = 42
A = STUDENT_ID % 5
B = (STUDENT_ID // 10) % 5
C = (STUDENT_ID // 100) % 5
N = 200 + 20 * A  # số bản ghi
MIN_FEATURES = 8 + B  # số thuộc tính đầu vào tối thiểu
MIN_OUTLIERS = 5 + 2 * C  # số bản ghi ngoại lai tối thiểu

N_GENUINE = 8  # ngoại lai hợp lệ (cực trị thật)
N_ERROR = 7  # ngoại lai lỗi dữ liệu
GENUINE_PLAN = [("whale", "HighValue", 3), ("reseller", "PromoHunter", 2),
                ("big_ticket", "Infrequent", 2), ("power_browser", "Browser", 1)]

# ----------------------------------------------------------------------------------------------
# 2. Lược đồ thuộc tính
# ----------------------------------------------------------------------------------------------
NUMERIC = ["age", "n_orders_12m", "avg_order_value", "total_spend_12m", "recency_days",
           "discount_usage_rate", "sessions_per_month", "avg_session_minutes", "pages_per_session",
           "cart_abandon_rate"]
CATEGORICAL = ["membership_tier", "device_type", "preferred_category", "app_installed"]
FEATURES = NUMERIC + CATEGORICAL  # 14 thuộc tính đầu vào (không có customer_id / nhãn)
ID_COL = "customer_id"
SKEWED = ["n_orders_12m", "avg_order_value", "total_spend_12m", "recency_days", "sessions_per_month",
          "avg_session_minutes", "pages_per_session"]

# miền giá trị hợp lệ (dùng để phát hiện lỗi ở các bước sau và để kiểm tra D0/D1)
VALID = {"age": (18, 100), "n_orders_12m": (1, 500), "avg_order_value": (0.01, 10000),
         "total_spend_12m": (0.01, 1e6), "recency_days": (0, 365), "discount_usage_rate": (0, 1),
         "sessions_per_month": (0, 100), "avg_session_minutes": (0, 240), "pages_per_session": (0, 100),
         "cart_abandon_rate": (0, 1)}

SEGMENTS = ["HighValue", "PromoHunter", "Infrequent", "Browser"]
SEG_WEIGHTS = np.array([48, 78, 66, 68], float)
DEVICES = ["mobile", "desktop", "tablet"]
CATS = ["fashion", "electronics", "home", "beauty", "grocery"]
TIERS = ["bronze", "silver", "gold"]

# ----------------------------------------------------------------------------------------------
# 3. Tham số của cơ chế sinh (hiệu chỉnh chỉ trên D0, xem docs/data_dictionary.md)
# ----------------------------------------------------------------------------------------------
# đặc điểm ẩn: E gắn bó | I ý định mua | W mức chi mỗi đơn | S nhạy cảm giá
_TRAIT_MEAN_BASE = np.array([[0.6, 1.0, 1.5, -0.5],     # HighValue
                             [0.1, 0.6, -0.1, 1.7],     # PromoHunter
                             [-1.6, 0.4, 0.3, 0.1],     # Infrequent
                             [0.9, -2.4, -0.35, -0.3]])  # Browser
# Hiệu chỉnh độ chồng lấn (chỉ dựa trên D0; xem docs/data_dictionary.md): khoảng cách giữa các phân khúc quanh
# trung bình chung nhân (E x1,0; I, W, S x1,5), phân tán trong cụm x0,6, nhiễu phương trình x0,75.
# Giữ E nguyên để sessions_per_month không tự tách được phân khúc. ARI K-Means trên D0 ~0,71.
TRAIT_GAP_SCALE = np.array([1.0, 1.5, 1.5, 1.5])
TRAIT_MEAN = _TRAIT_MEAN_BASE.mean(axis=0) + TRAIT_GAP_SCALE * (_TRAIT_MEAN_BASE - _TRAIT_MEAN_BASE.mean(axis=0))
TRAIT_SD = np.array([0.55, 0.60, 0.60, 0.60]) * 0.6
TRAIT_CORR = np.eye(4)
TRAIT_CORR[0, 1] = TRAIT_CORR[1, 0] = 0.20
TRAIT_CORR[1, 2] = TRAIT_CORR[2, 1] = 0.20
TRAIT_CORR[2, 3] = TRAIT_CORR[3, 2] = -0.35
AGE_MEAN = np.array([38.0, 29.0, 42.0, 26.0])
DEV_P = np.array([[.50, .40, .10], [.75, .15, .10], [.35, .55, .10], [.80, .10, .10]])
CAT_P = np.array([[.20, .35, .25, .10, .10], [.40, .10, .05, .35, .10],
                  [.10, .10, .30, .10, .40], [.35, .40, .10, .10, .05]])
P = dict(
    bridge_share=0.18, cat_soft=0.20, age_sd=9.0,
    dev_age_tilt=-0.04, cat_age_tilt=np.array([-0.2, 0.0, 0.2, -0.15, 0.15]),
    sess0=2.1, sess1=0.9, sess_sd=0.22,
    pg0=2.06, pg1=0.4, pg2=0.2, pg_sd=0.22, pg_cat=np.array([0.0, 0.15, 0.0, 0.0, -0.10]),
    mpp=1.2, mpp_dev=np.array([0.85, 1.15, 1.0]), min_sd=0.28,
    conv0=-2.6, conv1=1.0, conv_sd=0.18, nb_r=3.0,
    ab0=0.1, ab1=0.65, ab_sd=0.32,
    g0=-1.2, g1=1.6, g_sd=0.40,
    m0=3.95, m1=0.35, m2=0.5, aov_sd=0.22, aov_cat=np.array([0.0, 0.35, 0.15, -0.15, -0.5]),
    total_sd=0.10, rec_lapse=0.3, rec_shape=1.5,
    tier_noise=0.5, silver_cut=550.0, gold_cut=1400.0,
    app0=0.3, app1=0.8, app2=0.7,
)
# nhiễu D0 -> D1 (áp dụng cho mọi dòng)
NOISE = {"avg_order_value": ("mult", 0.12, (5.0, 500.0)),
         "recency_days": ("add", 4.0, (0, 365)),
         "avg_session_minutes": ("mult", 0.15, (1.0, 90.0)),
         "sessions_per_month": ("add", 1.5, (0.5, 60.0))}
# giá trị thiếu D2 -> D3: cột -> (cơ chế, tỉ lệ)
MISSING = {"age": ("MCAR", 0.08), "avg_session_minutes": ("MAR", 0.12), "preferred_category": ("MCAR", 0.05)}
MAR_WEIGHT = {"mobile": 0.07, "desktop": 0.18, "tablet": 0.15}  # xác suất thiếu tương đối theo device_type

log = logging.getLogger("gen")


# ----------------------------------------------------------------------------------------------
# 4. Tiện ích
# ----------------------------------------------------------------------------------------------
def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def _segment_sizes(n):
    raw = SEG_WEIGHTS / SEG_WEIGHTS.sum() * n
    sizes = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - sizes))[: n - sizes.sum()]:
        sizes[i] += 1
    return sizes


def _sample_categorical(rng, probs):
    cum = probs.cumsum(axis=1)
    idx = (rng.random(len(probs))[:, None] > cum).sum(axis=1)
    return np.minimum(idx, probs.shape[1] - 1)


def _tier_from_total(total, rng):
    score = np.log(np.maximum(total, 1.0)) + rng.normal(0, P["tier_noise"], np.shape(total))
    return np.where(score >= np.log(P["gold_cut"]), "gold", np.where(score >= np.log(P["silver_cut"]), "silver", "bronze"))


def _log_cells(stage, ids, column, before, after, itype, oclass="", note=""):
    return pd.DataFrame({"stage": stage, ID_COL: ids, "column": column, "injection_type": itype,
                         "outlier_class": oclass, "value_before": pd.Series(list(before), dtype=object),
                         "value_after": pd.Series(list(after), dtype=object), "note": note})


def _hash_frame(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values.tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# 5. D0: dữ liệu nền sạch
# ----------------------------------------------------------------------------------------------
def generate_clean(rng):
    sizes = _segment_sizes(N)
    seg = np.repeat(np.arange(4), sizes)
    n = len(seg)
    # khách "cầu nối": đặc điểm pha giữa phân khúc của mình và một phân khúc khác
    is_bridge = np.zeros(n, bool)
    is_bridge[rng.choice(n, int(round(P["bridge_share"] * n)), replace=False)] = True
    partner = (seg + rng.integers(1, 4, n)) % 4
    w = np.where(is_bridge, rng.uniform(0.35, 0.65, n), 1.0)

    def mix(table):
        t = np.asarray(table, float)
        return (w[:, None] if t.ndim == 2 else w) * t[seg] + ((1 - w)[:, None] if t.ndim == 2 else (1 - w)) * t[partner]

    # tầng 2: đặc điểm ẩn
    L = np.linalg.cholesky(np.outer(TRAIT_SD, TRAIT_SD) * TRAIT_CORR)
    E, I, W, S = (mix(TRAIT_MEAN) + rng.standard_normal((n, 4)) @ L.T).T

    age = np.clip(np.round(mix(AGE_MEAN) + rng.normal(0, P["age_sd"], n)), 18, 70)

    # tầng 3: thuộc tính quan sát được
    soft = P["cat_soft"]
    pd_dev = (1 - soft) * mix(DEV_P) + soft / 3
    pd_dev[:, 0] *= np.exp(P["dev_age_tilt"] * (age - 35))
    pd_dev /= pd_dev.sum(axis=1, keepdims=True)
    dev = _sample_categorical(rng, pd_dev)
    pd_cat = (1 - soft) * mix(CAT_P) + soft / 5
    pd_cat *= np.exp(P["cat_age_tilt"][None, :] * ((age - 35) / 10)[:, None])
    pd_cat /= pd_cat.sum(axis=1, keepdims=True)
    cat = _sample_categorical(rng, pd_cat)

    sessions = np.exp(P["sess0"] + P["sess1"] * E + rng.normal(0, P["sess_sd"], n))
    pages = np.exp(P["pg0"] + P["pg1"] * E - P["pg2"] * I + P["pg_cat"][cat] + rng.normal(0, P["pg_sd"], n))
    minutes = pages * P["mpp"] * P["mpp_dev"][dev] * np.exp(rng.normal(0, P["min_sd"], n))
    conv = _logistic(P["conv0"] + P["conv1"] * I + rng.normal(0, P["conv_sd"], n))
    mu_orders = 12.0 * sessions * conv
    r = P["nb_r"]
    orders = np.maximum(1, rng.negative_binomial(r, r / (r + mu_orders)))
    abandon = _logistic(P["ab0"] - P["ab1"] * I + rng.normal(0, P["ab_sd"], n))
    disc = _logistic(P["g0"] + P["g1"] * S + rng.normal(0, P["g_sd"], n))
    aov = np.exp(P["m0"] + P["m1"] * W + P["aov_cat"][cat] - P["m2"] * disc + rng.normal(0, P["aov_sd"], n))
    total = orders * aov * np.exp(rng.normal(-0.005, P["total_sd"], n))
    rec_mean = 365.0 / (orders + 1) * np.exp(-P["rec_lapse"] * E)
    rec = np.minimum(365, rng.gamma(P["rec_shape"], rec_mean / P["rec_shape"]))
    tier = _tier_from_total(total, rng)
    app = (rng.random(n) < _logistic(P["app0"] + P["app1"] * (dev == 0) + P["app2"] * E)).astype(int)

    df = pd.DataFrame({
        "age": age.astype(int),
        "n_orders_12m": orders.astype(int),
        "avg_order_value": np.round(np.clip(aov, 5.0, 500.0), 2),
        "total_spend_12m": np.round(total, 2),
        "recency_days": np.round(rec).astype(int),
        "discount_usage_rate": np.round(np.clip(disc, 0.01, 0.99), 3),
        "sessions_per_month": np.round(np.clip(sessions, 0.5, 60.0), 1),
        "avg_session_minutes": np.round(np.clip(minutes, 1.0, 90.0), 1),
        "pages_per_session": np.round(np.clip(pages, 1.0, 60.0), 1),
        "cart_abandon_rate": np.round(np.clip(abandon, 0.02, 0.98), 3),
        "membership_tier": tier,
        "device_type": np.array(DEVICES)[dev],
        "preferred_category": np.array(CATS)[cat],
        "app_installed": app,
    })
    truth = pd.DataFrame({
        "true_segment": np.array(SEGMENTS)[seg], "is_bridge": is_bridge,
        "partner_segment": np.where(is_bridge, np.array(SEGMENTS)[partner], ""),
        "bridge_weight_own": np.round(w, 3),
        "trait_E_engagement": np.round(E, 3), "trait_I_intent": np.round(I, 3),
        "trait_W_order_size": np.round(W, 3), "trait_S_price_sensitivity": np.round(S, 3)})
    perm = rng.permutation(n)  # xáo trộn để thứ tự dòng không lộ phân khúc
    df = df.iloc[perm].reset_index(drop=True)
    truth = truth.iloc[perm].reset_index(drop=True)
    ids = [f"C{i + 1:04d}" for i in range(n)]
    df.insert(0, ID_COL, ids)
    truth.insert(0, ID_COL, ids)
    return df[[ID_COL] + FEATURES], truth


# ----------------------------------------------------------------------------------------------
# 6. D1: nhiễu ngẫu nhiên
# ----------------------------------------------------------------------------------------------
def add_noise(d0, rng):
    d = d0.copy()
    logs = []
    for col, (kind, sd, (lo, hi)) in NOISE.items():
        old = d0[col].to_numpy().astype(float)
        if kind == "mult":
            new = old * np.exp(sd * rng.standard_normal(len(d)) - sd ** 2 / 2)  # kỳ vọng nhân = 1 (không thiên lệch)
        else:
            new = old + rng.normal(0, sd, len(d))
        new = np.clip(new, lo, hi)
        if col == "recency_days":
            new = np.round(new).astype(int)
        elif col in ("avg_order_value",):
            new = np.round(new, 2)
        else:
            new = np.round(new, 1)
        d[col] = new
        note = f"{'nhân lognormal' if kind == 'mult' else 'cộng Gaussian'} sd={sd}"
        logs.append(_log_cells("noise", d0[ID_COL], col, old, new, f"noise_{kind}", note=note))
    return d, pd.concat(logs, ignore_index=True)


# ----------------------------------------------------------------------------------------------
# 7. D2: ngoại lai
# ----------------------------------------------------------------------------------------------
def add_outliers(d1, truth, rng):
    d = d1.copy()
    seg = truth["true_segment"].to_numpy()
    avail = np.ones(len(d), bool)
    logs = []

    def pick(segname, k):
        pool = np.flatnonzero(avail & ((seg == segname) if segname else True))
        sel = rng.choice(pool, k, replace=False)
        avail[sel] = False
        return sel

    def setv(i, col, new, itype, oclass, note):
        before = d.at[i, col]
        d.at[i, col] = new
        logs.append(_log_cells("outlier", [d.at[i, ID_COL]], col, [before], [new], itype, oclass, note))

    def retotal(i, n_new, aov_new):
        return round(float(n_new * aov_new * np.exp(rng.normal(-0.005, P["total_sd"]))), 2)

    # cường độ đặt theo hàng rào Tukey (Q3 + 1,5 IQR) của chính D1, để ngoại lai thật sự là ngoại lai thống kê
    q1, q3 = d1[NUMERIC].quantile(0.25), d1[NUMERIC].quantile(0.75)
    fence_n = float((q3 + 1.5 * (q3 - q1))["n_orders_12m"])

    for itype, segname, k in GENUINE_PLAN:
        for i in pick(segname, k):
            n0, a0 = int(d.at[i, "n_orders_12m"]), float(d.at[i, "avg_order_value"])
            if itype == "whale":
                # tăng đơn vừa phải, tăng mạnh giá trị đơn; số đơn không quá 0,5 đơn/phiên để nhất quán với lượt truy cập
                s1 = round(min(float(d.at[i, "sessions_per_month"]) * rng.uniform(1.3, 1.8), 100.0), 1)
                n1 = max(n0, min(int(round(n0 * rng.uniform(1.3, 1.8))), int(0.5 * 12 * s1)))
                a1 = round(a0 * rng.uniform(2.0, 3.0), 2)
                setv(i, "n_orders_12m", n1, itype, "genuine", "khách cá voi: mua nhiều và đơn lớn")
                setv(i, "avg_order_value", a1, itype, "genuine", "khách cá voi")
                setv(i, "total_spend_12m", retotal(i, n1, a1), itype, "genuine", "tính lại theo n_orders x AOV")
                setv(i, "sessions_per_month", s1, itype, "genuine", "gắn bó cao")
                setv(i, "recency_days", int(min(d.at[i, "recency_days"], rng.integers(0, 8))), itype, "genuine", "mua rất gần đây")
            elif itype == "reseller":
                n1, a1 = int(round(fence_n * rng.uniform(1.5, 3.0))), round(float(rng.uniform(12, 30)), 2)
                s1 = round(min(max(float(d.at[i, "sessions_per_month"]), n1 / (12 * rng.uniform(0.5, 0.8))), 100.0), 1)
                setv(i, "n_orders_12m", n1, itype, "genuine", "đại lý mua sỉ: rất nhiều đơn nhỏ")
                setv(i, "avg_order_value", a1, itype, "genuine", "đại lý mua sỉ")
                setv(i, "total_spend_12m", retotal(i, n1, a1), itype, "genuine", "tính lại theo n_orders x AOV")
                setv(i, "sessions_per_month", s1, itype, "genuine", "nhiều phiên để đặt nhiều đơn (0,5-0,8 đơn/phiên)")
            elif itype == "big_ticket":
                n1, a1 = int(rng.integers(1, 3)), round(float(rng.uniform(1500, 3000)), 2)
                setv(i, "n_orders_12m", n1, itype, "genuine", "mua một lần món giá trị lớn")
                setv(i, "avg_order_value", a1, itype, "genuine", "món giá trị lớn (đồ gia dụng, điện tử)")
                tot = retotal(i, n1, a1)
                setv(i, "total_spend_12m", tot, itype, "genuine", "tính lại theo n_orders x AOV")
                setv(i, "membership_tier", str(_tier_from_total(np.array([tot]), rng)[0]), itype, "genuine", "hạng suy từ tổng chi")
            else:  # power_browser
                setv(i, "sessions_per_month", round(float(rng.uniform(48, 60)), 1), itype, "genuine", "người so sánh giá kỹ lưỡng")
                setv(i, "pages_per_session", round(float(rng.uniform(38, 55)), 1), itype, "genuine", "xem rất nhiều trang")
                setv(i, "avg_session_minutes", round(float(rng.uniform(50, 80)), 1), itype, "genuine", "phiên rất dài")
            if itype in ("whale",):
                setv(i, "membership_tier", str(_tier_from_total(np.array([d.at[i, "total_spend_12m"]]), rng)[0]),
                     itype, "genuine", "hạng suy từ tổng chi")

    errors = [("age", 150, "typo_age", "nhập sai năm sinh"),
              ("recency_days", -12, "negative_recency", "lệch đồng hồ hệ thống"),
              ("discount_usage_rate", 1.8, "rate_gt_1", "số lượt dùng mã lớn hơn số đơn"),
              ("avg_order_value", None, "unit_x100", "nhầm đơn vị tiền (x100)"),
              ("avg_session_minutes", 1440.0, "session_not_closed", "phiên không đóng, treo qua đêm"),
              ("cart_abandon_rate", 1.35, "rate_gt_1", "chia sai mẫu số"),
              ("sessions_per_month", 999.0, "sentinel_999", "mã sentinel cho 'không có dữ liệu'")]
    assert len(errors) == N_ERROR
    for col, val, itype, note in errors:
        i = pick(None, 1)[0]
        new = round(float(d.at[i, "avg_order_value"]) * 100, 2) if col == "avg_order_value" else val
        setv(i, col, new, itype, "error", note)
    return d, pd.concat(logs, ignore_index=True)


# ----------------------------------------------------------------------------------------------
# 8. D3: giá trị thiếu
# ----------------------------------------------------------------------------------------------
def add_missing(d2, outlier_log, rng):
    d = d2.copy()
    protected = set(zip(outlier_log[ID_COL], outlier_log["column"]))  # không xóa ô đã mang ngoại lai
    logs = []
    ids = d[ID_COL].to_numpy()
    for col, (mech, rate) in MISSING.items():
        k = int(round(rate * len(d)))
        elig = np.array([i for i in range(len(d)) if (ids[i], col) not in protected])
        if mech == "MAR":
            wts = d["device_type"].map(MAR_WEIGHT).to_numpy()[elig]
            sel = rng.choice(elig, k, replace=False, p=wts / wts.sum())
        else:
            sel = rng.choice(elig, k, replace=False)
        sel = np.sort(sel)
        before = d.loc[sel, col].tolist()
        if col == "age":
            d[col] = d[col].astype("Int64")
        d.loc[sel, col] = pd.NA if col == "age" else np.nan
        logs.append(_log_cells("missing", ids[sel], col, before, [np.nan] * len(sel), f"missing_{mech}", note=f"{mech} {rate:.0%}"))
    d["age"] = d["age"].astype("Int64")
    return d, pd.concat(logs, ignore_index=True)


# ----------------------------------------------------------------------------------------------
# 9. Điều phối và ghi file
# ----------------------------------------------------------------------------------------------
def generate_all(seed=SEED):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    d0, truth = generate_clean(streams[0])
    d1, noise_log = add_noise(d0, streams[1])
    d2, out_log = add_outliers(d1, truth, streams[2])
    d3, miss_log = add_missing(d2, out_log, streams[3])
    inj = pd.concat([noise_log, out_log, miss_log], ignore_index=True)
    return dict(d0=d0, d1=d1, d2=d2, d3=d3, truth=truth, inj=inj)


DICT_SPEC = [
    ("age", "Nhân khẩu", "Tuổi khách", "int", "18-70", "Chuẩn theo phân khúc, cắt biên", "Đặc điểm khách; phân biệt yếu (thuộc tính ít thông tin)", "MCAR ~8%"),
    ("n_orders_12m", "Mua hàng", "Số đơn 12 tháng gần nhất", "int", ">=1 (thực tế 1-60)", "Nhị thức âm, lệch phải; mean = 12 x sessions x tỉ lệ chuyển đổi", "Tần suất giao dịch", ""),
    ("avg_order_value", "Chi tiêu", "Giá trị đơn trung bình (USD)", "float", ">0 (thực tế 8-300)", "Lognormal, lệch phải", "Quy mô mỗi lần mua", ""),
    ("total_spend_12m", "Chi tiêu", "Tổng chi 12 tháng (USD)", "float", ">0", "Lệch phải mạnh; ~ n_orders x AOV", "Giá trị khách; cố ý đa cộng tuyến với n_orders", ""),
    ("recency_days", "Mua hàng", "Số ngày từ đơn gần nhất", "int", "0-365", "Gamma, lệch phải; trung bình ~ 365/(n_orders+1)", "Mức còn hoạt động", ""),
    ("discount_usage_rate", "Khuyến mãi", "Tỉ lệ đơn dùng mã giảm giá", "float", "[0,1]", "Logistic của độ nhạy giá", "Nhóm săn khuyến mãi", ""),
    ("sessions_per_month", "Tương tác", "Số phiên truy cập mỗi tháng", "float", ">=0.5", "Lognormal theo mức gắn bó", "Mức gắn bó với nền tảng", ""),
    ("avg_session_minutes", "Tương tác", "Thời lượng phiên trung bình (phút)", "float", "1-90", "Lognormal; ~ pages x phút/trang x hệ số thiết bị", "Mức tương tác", "MAR theo device_type ~12%"),
    ("pages_per_session", "Duyệt", "Số trang sản phẩm xem mỗi phiên", "float", "1-60", "Lognormal; tăng theo gắn bó, giảm theo ý định mua", "Hành vi duyệt", ""),
    ("cart_abandon_rate", "Duyệt", "Tỉ lệ giỏ hàng bị bỏ", "float", "[0,1]", "Logistic, giảm theo ý định mua", "Duyệt nhiều nhưng không chốt đơn", ""),
    ("membership_tier", "Phân loại thứ tự", "Hạng thành viên", "ordinal", "bronze<silver<gold", "Cắt ngưỡng của log(total_spend)+nhiễu", "Hạng đạt được nhờ chi tiêu", ""),
    ("device_type", "Phân loại định danh", "Thiết bị chính", "nominal", "mobile/desktop/tablet", "Multinomial theo phân khúc, tuổi", "Hành vi theo thiết bị", ""),
    ("preferred_category", "Phân loại định danh", "Ngành hàng ưa thích", "nominal", "fashion/electronics/home/beauty/grocery", "Multinomial theo phân khúc, tuổi", "Sở thích mua", "MCAR ~5%"),
    ("app_installed", "Nhị phân", "Đã cài ứng dụng", "binary", "0/1", "Bernoulli, tăng theo mobile và gắn bó", "Mức gắn bó kênh", ""),
]


def build_dictionary(d3):
    rows = []
    for name, grp, meaning, dtype, domain, dist, why, miss in DICT_SPEC:
        s = d3[name]
        row = dict(attribute=name, group=grp, meaning=meaning, dtype=dtype, valid_domain=domain,
                   expected_distribution=dist, rationale=why,
                   noise_added=("yes" if name in NOISE else "no"), missing_mechanism=miss,
                   n_missing_final=int(s.isna().sum()), pct_missing_final=round(100 * s.isna().mean(), 1))
        if name in NUMERIC:
            x = pd.to_numeric(s, errors="coerce")
            row.update(min_final=x.min(), median_final=x.median(), max_final=x.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(res, out_dir):
    gen = out_dir / "01_generated"
    hid = out_dir / "hidden"
    gen.mkdir(parents=True, exist_ok=True)
    hid.mkdir(parents=True, exist_ok=True)
    names = {"d0": "data_clean_reference.csv", "d1": "data_after_noise.csv",
             "d2": "data_after_outliers.csv", "d3": "data_raw_corrupted.csv"}
    for k, fn in names.items():
        res[k].to_csv(gen / fn, index=False)
    build_dictionary(res["d3"]).to_csv(gen / "data_dictionary.csv", index=False)
    res["truth"].to_csv(hid / "ground_truth.csv", index=False)
    res["inj"].to_csv(hid / "injection_log.csv", index=False)
    return gen, hid, names


# ----------------------------------------------------------------------------------------------
# 10. Kiểm tra nghiệm thu
# ----------------------------------------------------------------------------------------------
class Checker:
    def __init__(self):
        self.rows = []

    def __call__(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        log.info("[%s] %s %s", "PASS" if ok else "FAIL", name, detail)

    @property
    def all_ok(self):
        return all(ok for _, ok, _ in self.rows)


def _same(a, b, cols):
    for c in cols:
        x, y = a[c], b[c]
        if c in NUMERIC or c == "app_installed":
            if not np.allclose(x.astype(float), y.astype(float), equal_nan=True, atol=1e-9):
                return False
        elif not x.equals(y):
            return False
    return True


def _violations(df):
    bad = pd.Series(False, index=df.index)
    for c, (lo, hi) in VALID.items():
        v = df[c].astype(float)
        bad |= (v < lo) | (v > hi)
    return bad


def _identity_dev(df):
    return np.abs(np.log(df["total_spend_12m"].astype(float) / (df["n_orders_12m"].astype(float) * df["avg_order_value"].astype(float))))


def _design_matrix(df):
    x = df[NUMERIC].astype(float).copy()
    x[SKEWED] = np.log1p(x[SKEWED])
    x["membership_tier"] = df["membership_tier"].map({"bronze": 0, "silver": 1, "gold": 2})
    x["app_installed"] = df["app_installed"].astype(float)
    X = pd.concat([x, pd.get_dummies(df[["device_type", "preferred_category"]]).astype(float)], axis=1)
    return ((X - X.mean()) / X.std(ddof=0)).to_numpy()


def validate(res, out_dir, files):
    ck = Checker()
    d0, d1, d2, d3, truth, inj = (res[k] for k in ("d0", "d1", "d2", "d3", "truth", "inj"))
    gen = out_dir / "01_generated"

    log.info("=" * 78)
    log.info("A. RÀNG BUỘC CỦA ĐỀ (MSSV %d: a=%d b=%d c=%d)", STUDENT_ID, A, B, C)
    log.info("=" * 78)
    ck("N = 200+20a", all(len(d) == N for d in (d0, d1, d2, d3)), f"N={N}, thực tế {[len(d) for d in (d0, d1, d2, d3)]}")
    ck("Số thuộc tính >= 8+b", len(FEATURES) >= MIN_FEATURES, f"{len(FEATURES)} >= {MIN_FEATURES}")
    miss_cols = [c for c in FEATURES if d3[c].isna().any()]
    ck("Thiếu ở >= 2 thuộc tính", len(miss_cols) >= 2, f"{miss_cols}")
    noisy_cols = [c for c in NUMERIC if not _same(d0, d1, [c])]
    ck("Nhiễu ở >= 2 thuộc tính số", len(noisy_cols) >= 2, f"{noisy_cols}")
    out_ids = inj.loc[inj["stage"] == "outlier", ID_COL].unique()
    ck("Ngoại lai >= 5+2c bản ghi khác nhau", len(out_ids) >= MIN_OUTLIERS, f"{len(out_ids)} >= {MIN_OUTLIERS}")
    ck("Lưu đủ 4 file đối chiếu", all((gen / fn).exists() for fn in files.values()), ", ".join(files.values()))

    log.info("-" * 78)
    log.info("B. TOÀN VẸN GIỮA CÁC PHIÊN BẢN")
    ck("customer_id duy nhất, cùng thứ tự ở D0-D3", d0[ID_COL].is_unique and all(d[ID_COL].equals(d0[ID_COL]) for d in (d1, d2, d3)))
    ck("Cột D0-D3 chỉ gồm id + 14 thuộc tính, không có nhãn/cờ", all(list(d.columns) == [ID_COL] + FEATURES for d in (d0, d1, d2, d3)))
    leak = {"true_segment", "is_bridge", "partner_segment", "outlier", "is_outlier"} & set(d3.columns)
    ck("Không có thông tin nhãn/outlier trong feature matrix", not leak and ID_COL not in FEATURES)
    ck("D0, D1, D2 không có ô trống", not d0.isna().any().any() and not d1.isna().any().any() and not d2.isna().any().any())
    ck("D0 -> D1: chỉ 4 cột nhiễu đổi, các cột khác giữ nguyên", sorted(noisy_cols) == sorted(NOISE) and
       _same(d0, d1, [c for c in FEATURES if c not in NOISE]))
    ck("D0 và D1 nằm trong miền hợp lệ", not _violations(d0).any() and not _violations(d1).any())
    changed12 = [i for i in range(N) if not _same(d1.iloc[[i]], d2.iloc[[i]], FEATURES)]
    ck("D1 -> D2: chỉ đổi đúng các dòng ngoại lai", set(d1.loc[changed12, ID_COL]) == set(out_ids), f"{len(changed12)} dòng đổi")
    err = inj[(inj["stage"] == "outlier") & (inj["outlier_class"] == "error")]
    gen_ids = inj.loc[(inj["stage"] == "outlier") & (inj["outlier_class"] == "genuine"), ID_COL].unique()
    ck("Ngoại lai lỗi: mỗi khách hỏng đúng 1 ô", err.groupby(ID_COL).size().eq(1).all() and len(err) == N_ERROR)
    ck("Ngoại lai hợp lệ + lỗi = tổng, không trùng khách", len(gen_ids) == N_GENUINE and not (set(gen_ids) & set(err[ID_COL])))
    mask = d3.isna()
    allowed_miss = set(MISSING)
    ck("D2 -> D3: thiếu chỉ ở 3 cột đã định", set(d3.columns[mask.any()]) <= allowed_miss, f"{sorted(d3.columns[mask.any()])}")
    same_obs = all(d2.loc[~mask[c], c].astype(float).equals(d3.loc[~mask[c], c].astype(float)) if c in NUMERIC
                   else d2.loc[~mask[c], c].equals(d3.loc[~mask[c], c]) for c in FEATURES)
    ck("D3: ô không thiếu giữ nguyên giá trị D2", same_obs)
    prot = set(zip(inj.loc[inj["stage"] == "outlier", ID_COL], inj.loc[inj["stage"] == "outlier", "column"]))
    id_to_pos = {v: i for i, v in enumerate(d3[ID_COL])}
    ck("Không xóa ô đã mang ngoại lai", not any(mask.iloc[id_to_pos[i]][c] for i, c in prot))
    exact = all(int(mask[c].sum()) == int(round(MISSING[c][1] * N)) for c in MISSING)
    ck("Số ô thiếu khớp thiết kế", exact, {c: int(mask[c].sum()) for c in MISSING})

    log.info("-" * 78)
    log.info("C. NGOẠI LAI: PHÁT HIỆN ĐƯỢC ĐÚNG NHƯ THIẾT KẾ")
    q1, q3 = d1[NUMERIC].quantile(0.25), d1[NUMERIC].quantile(0.75)
    iqr = q3 - q1
    hi15, hi3 = q3 + 1.5 * iqr, q3 + 3 * iqr
    def beyond(i, k):
        row = d2.iloc[i][NUMERIC].astype(float)
        return bool(((row > q3 + k * iqr) | (row < q1 - k * iqr)).any())
    gpos = [id_to_pos[i] for i in gen_ids]
    ck("Mọi ngoại lai hợp lệ vượt hàng rào Tukey 1,5 IQR (>=1 cột)", all(beyond(i, 1.5) for i in gpos))
    ops = d2["n_orders_12m"].astype(float) / (12 * d2["sessions_per_month"].astype(float))
    ck("Ngoại lai hợp lệ nhất quán nghiệp vụ: số đơn <= 1 đơn/phiên (n_orders <= 12 x sessions)",
       all(ops.iloc[i] <= 1.0 for i in gpos), f"cao nhất {max(ops.iloc[i] for i in gpos):.2f}")
    log.info("      (vượt hàng rào 3 IQR: %d/%d)", sum(beyond(i, 3.0) for i in gpos), len(gpos))
    viol = _violations(d2)
    ident = _identity_dev(d2) > 0.7
    err_pos = {c: id_to_pos[err.loc[err["column"] == c, ID_COL].iloc[0]] for c in err["column"]}
    ck("Lỗi vi phạm miền hợp lệ (6 lỗi) hoặc phá đẳng thức total ~ n x AOV (lỗi x100)",
       all(viol.iloc[p] for c, p in err_pos.items() if c != "avg_order_value") and bool(ident.iloc[err_pos["avg_order_value"]]))
    ck("Không dòng nào khác vi phạm miền hợp lệ/đẳng thức", set(np.flatnonzero(viol | ident)) == set(err_pos.values()),
       f"vi phạm miền: {int(viol.sum())}, vi phạm đẳng thức: {int(ident.sum())}")
    log.info("      %d dòng ngoại lai vượt hàng rào Tukey 1,5 IQR ở D2 tổng cộng (gồm cả đuôi tự nhiên)", sum(beyond(i, 1.5) for i in range(N)))

    log.info("-" * 78)
    log.info("D. NHIỄU: ĐỘ LỚN VÀ TÍNH KHÔNG THIÊN LỆCH")
    for c in NOISE:
        delta = d1[c].astype(float) - d0[c].astype(float)
        snr = d0[c].astype(float).var() / max(delta.var(), 1e-12)
        rel = (d1[c].astype(float) / d0[c].astype(float)).mean()
        log.info("      %-22s tỉ lệ dòng đổi=%.0f%%  SNR(var gốc/var nhiễu)=%.1f  trung bình D1/D0=%.3f  sd nhiễu=%.2f",
                 c, 100 * (delta != 0).mean(), snr, rel, delta.std())
    ck("Nhiễu không thiên lệch (|mean D1 / mean D0 - 1| < 3% ở cả 4 cột)",
       all(abs(d1[c].astype(float).mean() / d0[c].astype(float).mean() - 1) < 0.03 for c in NOISE),
       {c: round(d1[c].astype(float).mean() / d0[c].astype(float).mean() - 1, 4) for c in NOISE})
    ck("Nhiễu đủ rõ (>= 70% dòng thay đổi ở mỗi cột nhiễu)", all((d1[c] != d0[c]).mean() >= 0.7 for c in NOISE))

    log.info("-" * 78)
    log.info("E. CẤU TRÚC D0 (cơ chế nghiệp vụ và độ chồng lấn)")
    L = d0.copy()
    cor = lambda a, b: float(np.corrcoef(np.log1p(L[a].astype(float)), np.log1p(L[b].astype(float)))[0, 1])
    c1, c2 = cor("n_orders_12m", "total_spend_12m"), cor("sessions_per_month", "pages_per_session")
    c3 = float(np.corrcoef(L["discount_usage_rate"], L["avg_order_value"])[0, 1])
    c4 = float(L["membership_tier"].map({"bronze": 0, "silver": 1, "gold": 2}).corr(L["total_spend_12m"], method="spearman"))
    ck("corr(log n_orders, log total_spend) >= 0,8", c1 >= 0.8, f"{c1:.2f}")
    ck("corr(log sessions, log pages) >= 0,4", c2 >= 0.4, f"{c2:.2f}")
    ck("corr(discount, AOV) < 0", c3 < 0, f"{c3:.2f}")
    ck("Spearman(tier, total_spend) >= 0,5", c4 >= 0.5, f"{c4:.2f}")
    ops0 = L["n_orders_12m"].astype(float) / (12 * L["sessions_per_month"].astype(float))
    ck("D0 nhất quán nghiệp vụ: số đơn <= 1 đơn/phiên ở mọi khách", bool((ops0 <= 1.0).all()), f"cao nhất {ops0.max():.2f}")
    sk = L[NUMERIC].astype(float).skew()
    ck("Phân phối lệch phải (skew > 1: total_spend, n_orders)", sk["total_spend_12m"] > 1 and sk["n_orders_12m"] > 1,
       f"skew total={sk['total_spend_12m']:.2f}, n_orders={sk['n_orders_12m']:.2f}")
    sizes = truth["true_segment"].value_counts()
    ck("4 phân khúc, kích thước 48/78/66/68", sorted(sizes.values.tolist()) == [48, 66, 68, 78], dict(sizes))
    ck("Khách cầu nối 15-21%", 0.15 <= truth["is_bridge"].mean() <= 0.21, f"{truth['is_bridge'].mean():.1%}")
    ck("Thứ tự dòng không lộ phân khúc (phân khúc không nhóm theo id)",
       truth["true_segment"].ne(truth["true_segment"].shift()).mean() > 0.5)
    log.info("      Trung vị D0 theo phân khúc (chỉ để kiểm tra thiết kế; nhãn lấy từ data/hidden):")
    med = L.assign(seg=truth["true_segment"].values).groupby("seg")[NUMERIC].median().round(2).loc[SEGMENTS]
    for line in med.to_string().splitlines():
        log.info("        %s", line)

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.tree import DecisionTreeClassifier
        X = _design_matrix(d0)
        y = truth["true_segment"].to_numpy()
        aris, nmis, sils = [], [], []
        for r in range(5):
            lab = KMeans(4, n_init=10, random_state=r).fit_predict(X)
            aris.append(adjusted_rand_score(y, lab)); nmis.append(normalized_mutual_info_score(y, lab)); sils.append(silhouette_score(X, lab))
        ari = float(np.mean(aris))
        log.info("      K-Means k=4 trên D0 (seed %d): ARI=%.2f NMI=%.2f Silhouette=%.2f (TB 5 seed K-Means)", SEED, ari, np.mean(nmis), np.mean(sils))
        ck("D0 seed chính: ARI K-Means trong 0,60-0,85 (dung sai do mẫu nhỏ)", 0.60 <= ari <= 0.85, f"{ari:.2f}")
        # thuộc tính của thiết kế: trung bình trên 10 seed sinh dữ liệu mới (không dùng khi hiệu chỉnh)
        fresh = []
        for s in range(200, 210):
            dd, tt = generate_clean(np.random.default_rng(np.random.SeedSequence(s).spawn(4)[0]))
            fresh.append(adjusted_rand_score(tt["true_segment"], KMeans(4, n_init=10, random_state=0).fit_predict(_design_matrix(dd))))
        log.info("      ARI trên 10 seed sinh mới (200-209): %.2f ± %.2f", np.mean(fresh), np.std(fresh))
        ck("Thiết kế: ARI trung bình 10 seed mới trong khoảng 0,70-0,80", 0.70 <= np.mean(fresh) <= 0.80, f"{np.mean(fresh):.2f}")
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        accs = {c: cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), d0[[c]].astype(float), y, cv=cv).mean()
                for c in NUMERIC}
        best = max(accs, key=accs.get)
        log.info("      Độ chính xác cây quyết định 1 thuộc tính (depth 3, CV5): %s",
                 ", ".join(f"{k}={v:.2f}" for k, v in sorted(accs.items(), key=lambda t: -t[1])))
        ck("Không thuộc tính đơn lẻ nào tách được phân khúc (< 0,70)", accs[best] < 0.70, f"cao nhất {best}={accs[best]:.2f}")
    except ImportError:
        log.warning("Không có scikit-learn: bỏ qua kiểm tra ARI và cây quyết định một thuộc tính")

    log.info("-" * 78)
    log.info("F. TÁI LẬP VÀ FILE ĐÃ LƯU")
    res2 = generate_all(SEED)
    ck("Chạy lại cùng seed cho kết quả giống hệt (hash D0-D3, truth, log)",
       all(_hash_frame(res[k]) == _hash_frame(res2[k]) for k in res), " ".join(f"{k}:{_hash_frame(res[k])[:8]}" for k in ("d0", "d1", "d2", "d3")))
    for k, fn in files.items():
        back = pd.read_csv(gen / fn)
        ok = list(back.columns) == list(res[k].columns) and len(back) == N
        for c in NUMERIC:
            ok &= np.allclose(back[c].astype(float), res[k][c].astype(float), equal_nan=True, atol=1e-6)
        ck(f"Đọc lại {fn} khớp bộ nhớ", ok)

    log.info("-" * 78)
    log.info("G. THỐNG KÊ D3 (đầu vào của các câu sau)")
    m = d3.isna().sum()
    log.info("      Giá trị thiếu: %s", {c: f"{int(n)} ({100 * n / N:.1f}%)" for c, n in m[m > 0].items()})
    dev_rate = d3.assign(_m=d3["avg_session_minutes"].isna()).groupby("device_type")["_m"].mean().round(3).to_dict()
    log.info("      Tỉ lệ thiếu avg_session_minutes theo device_type (MAR): %s", dev_rate)
    desc = d3[NUMERIC].astype(float).describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)
    desc["skew"] = d3[NUMERIC].astype(float).skew().round(2)
    for line in desc.to_string().splitlines():
        log.info("        %s", line)
    for c in CATEGORICAL:
        log.info("      %s: %s", c, d3[c].value_counts(dropna=False).to_dict())
    log.info("=" * 78)
    n_ok = sum(ok for _, ok, _ in ck.rows)
    log.info("KẾT QUẢ: %d/%d kiểm tra đạt", n_ok, len(ck.rows))
    return ck


# ----------------------------------------------------------------------------------------------
def _root():
    try:
        here = Path(__file__).resolve().parent
        return here.parent if here.name == "src" else here
    except NameError:  # chạy trong notebook
        return Path.cwd()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=None, help="thư mục data (mặc định <repo>/data)")
    ap.add_argument("--log-dir", type=Path, default=None, help="thư mục logs (mặc định <repo>/logs)")
    args, _ = ap.parse_known_args(argv)  # bỏ qua tham số do Jupyter/Colab truyền vào
    out_dir = args.out_dir or _root() / "data"
    log_dir = args.log_dir or _root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "01_generate_data.log", mode="w", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers, force=True)

    log.info("Sinh dữ liệu: seed=%d, N=%d, đặc trưng=%d, ngoại lai=%d", SEED, N, len(FEATURES), N_GENUINE + N_ERROR)
    assert N_GENUINE + N_ERROR >= MIN_OUTLIERS, "số ngoại lai thiết kế < 5+2c"
    res = generate_all(SEED)
    gen, hid, files = write_outputs(res, out_dir)
    log.info("Đã ghi: %s", ", ".join(files.values()) + ", data_dictionary.csv")
    log.info("Đã ghi (ẩn): ground_truth.csv, injection_log.csv -> %s", hid)
    ck = validate(res, out_dir, files)
    return 0 if ck.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
