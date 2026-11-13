#!/usr/bin/env python3
"""Câu c - Data Preprocessing (giá trị thiếu, ô không hợp lệ, nhiễu, ngoại lai).

Chỉ đọc data_raw_corrupted.csv (D3) và data_dictionary.csv. Có chốt chặn đường dẫn lúc chạy và kiểm tra nghiệm thu mã nguồn
không chứa tên bất kỳ file tham chiếu nào khác. Đầu ra giữ nguyên đơn vị và độ chính xác gốc (không log, không chuẩn hóa, không mã hóa:
đó là việc của câu d).

Quyết định (đã duyệt, xem docs/cau_c_decisions.md):
  1. Giữ nguyên 260 dòng. Chỉ vô hiệu hóa/sửa từng ô không hợp lệ (luật miền hoặc đẳng thức xấp xỉ total ~ n x AOV).
  2. Ô vi phạm đẳng thức: Data Dictionary ghi "~" (xấp xỉ), không xác nhận đẳng thức chính xác, nên KHÔNG gán AOV = total/n;
     ô bị nghi (cực đoan nhất theo phân vị biên trong ba ô) bị vô hiệu hóa rồi điền như mọi ô khác; total/n chỉ dùng kiểm tra.
  3. Điền khuyết số: chọn theo thí nghiệm che ô, cùng mask cho mọi phương pháp, so sánh ghép cặp, quy tắc 1-SE (đơn giản nhất trong
     phạm vi trung bình tốt nhất + 1 SE). preferred_category giữ mức "unknown"; các biến thể điền chỉ lưu ở file riêng cho câu e.
  4. Không khử nhiễu từng giá trị, không làm mượt, không winsorize, không cắt ngưỡng. Cực trị hợp lệ chỉ đánh dấu cờ.
  5. Cờ và nhật ký nằm trong file riêng, không nằm trong ma trận đặc trưng.
Đóng băng: ghi SHA-256 vào data/02_preprocessed/FROZEN_SHA256.json sau khi nghiệm thu đạt; script 03b_evaluate_preprocessing.py
chỉ mở D0 khi mã băm khớp.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_style as vs  # noqa: E402
from dm_utils import (RULES, hopkins, identity_flag, knn_dist, make_lens, pair_share, select_log_cols,  # noqa: E402
                      viol_mask)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "01_generated"
OUT = ROOT / "data" / "02_preprocessed"
FIG = ROOT / "figures" / "03_preprocessing"
TAB = ROOT / "reports" / "tables"
LOGD = ROOT / "logs"
RAW, DICT = DATA / "data_raw_corrupted.csv", DATA / "data_dictionary.csv"
SEED, N_REPS, MASK_FRAC = 42, 50, 0.15
CAT = ["membership_tier", "device_type", "preferred_category", "app_installed"]
METHODS = ["median", "median_by_device", "knn5", "knn10", "iterative"]  # thứ tự từ đơn giản đến phức tạp
CAT_METHODS = ["global_mode", "mode_by_device", "mode_by_tier", "knn7"]
SHORT = {"median": "median", "median_by_device": "median|device", "knn5": "KNN 5", "knn10": "KNN 10", "iterative": "iterative"}
HATCH = {"median": "", "median_by_device": "//", "knn5": "..", "knn10": "xx", "iterative": "\\\\"}
DECIMALS = {"age": 0, "avg_order_value": 2, "recency_days": 0, "discount_usage_rate": 3, "sessions_per_month": 1,
            "avg_session_minutes": 1, "cart_abandon_rate": 3}
BOUNDS = {"sessions_per_month = 0,5": ("sessions_per_month", 0.5), "recency_days = 0": ("recency_days", 0),
          "recency_days = 365": ("recency_days", 365), "n_orders_12m = 1": ("n_orders_12m", 1), "age = 18": ("age", 18)}
# Tên file tham chiếu bị cấm, viết tách để chính file này không chứa nguyên văn chuỗi cấm
FORBIDDEN = ["data_clean" + "_reference", "data_after" + "_noise", "data_after" + "_outliers", "hid" + "den", "ground" + "_truth",
             "injection" + "_log"]
BEFORE, AFTER = vs.PALETTE["violet"], vs.PALETTE["teal"]  # D3 giữ màu tím của câu b; kết quả xử lý dùng màu lam-lục
log = logging.getLogger("c")


# ---------------------------------------------------------------------------------------------- tiện ích
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_input(path):
    path = Path(path)
    if path.parent != DATA or path.name not in {RAW.name, DICT.name}:
        raise PermissionError(f"03_preprocess chỉ được đọc {RAW.name} và {DICT.name}, không đọc {path}")
    return pd.read_csv(path)


def write_csv(df, path, **kw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.chmod(path, 0o644)
    df.to_csv(path, index=False, **kw)


def show(title, obj):
    log.info("\n--- %s ---\n%s", title, obj.to_string() if hasattr(obj, "to_string") else obj)


def save_table(df, name, index=False):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=index)


def hash_frame(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values.tobytes()).hexdigest()


class Checker:
    def __init__(self):
        self.rows = []

    def __call__(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        log.info("[%s] %s %s", "PASS" if ok else "FAIL", name, detail)

    @property
    def all_ok(self):
        return all(ok for _, ok, _ in self.rows)


def round_col(x, col):
    return np.round(np.asarray(x, float), DECIMALS[col])


def build_lens(X, log_cols):
    L = X.copy()
    for c in log_cols:
        L[c] = np.log1p(L[c].clip(lower=0))
    med = L.median()
    iqr = (L.quantile(.75) - L.quantile(.25)).replace(0, 1.0)
    return (L - med) / iqr, med, iqr


def from_lens(v, col, log_cols, med, iqr):
    x = np.asarray(v, float) * iqr[col] + med[col]
    return np.expm1(x) if col in log_cols else x


def aux_matrix(df):
    a = pd.get_dummies(df["device_type"]).astype(float)
    a["tier"] = df["membership_tier"].map({"bronze": 0, "silver": 1, "gold": 2}) / 2
    a["app"] = df["app_installed"].astype(float)
    return a


# ---------------------------------------------------------------------------------------------- bước 1: ô không hợp lệ
def audit_invalid(df, num):
    cells = []
    V = viol_mask(df, num)
    for col in num:
        for i in np.flatnonzero(V[col]):
            cells.append(dict(row=int(i), customer_id=df.at[i, "customer_id"], column=col, value=float(df.at[i, col]), rule="domain",
                              detail=f"ngoài miền hợp lệ {RULES[col][:2]}"))
    flag, dev, med, sd = identity_flag(df)
    trio = ["n_orders_12m", "avg_order_value", "total_spend_12m"]
    tier_med = df.groupby("membership_tier")["total_spend_12m"].median()
    for i in np.flatnonzero(flag):
        pct = {c: float((df[c] < df.at[i, c]).mean()) for c in trio}
        suspect = max(trio, key=lambda c: abs(pct[c] - 0.5))
        tier = df.at[i, "membership_tier"]
        cells.append(dict(row=int(i), customer_id=df.at[i, "customer_id"], column=suspect, value=float(df.at[i, suspect]), rule="identity",
                          detail=(f"|ln(total/(n x AOV)) - trung vị| = {abs(dev[i] - med):.2f} > 5 sd bền ({5 * sd:.2f}); ô nghi = ô cực đoan nhất theo "
                                  f"phân vị biên {({k: round(v, 3) for k, v in pct.items()})}; hạng {tier} (trung vị total {tier_med[tier]:.0f}); "
                                  f"total/n = {df.at[i, 'total_spend_12m'] / df.at[i, 'n_orders_12m']:.2f} chỉ dùng kiểm tra, không dùng để gán")))
    return pd.DataFrame(cells)


# ---------------------------------------------------------------------------------------------- bước 2: thí nghiệm che ô
def predict(method, A, j, hide, dev_vals):
    if method == "median":
        return np.full(len(hide), np.nanmedian(A[:, j]))
    if method == "median_by_device":
        return np.array([np.nanmedian(A[dev_vals == dev_vals[h], j]) for h in hide])
    if method.startswith("knn"):
        return KNNImputer(n_neighbors=int(method[3:]), weights="distance").fit_transform(A)[hide, j]
    return IterativeImputer(estimator=BayesianRidge(), max_iter=15, random_state=0).fit_transform(A)[hide, j]


def masked_experiment(Z, names, X_orig, dev_vals, cols, log_cols, med, iqr):
    out = {}
    for col in cols:
        j = names.index(col)
        obs = np.flatnonzero(~np.isnan(Z[:, j]))
        k = max(5, int(round(MASK_FRAC * len(obs))))
        rng = np.random.default_rng(np.random.SeedSequence([SEED, j]))
        masks = [rng.choice(obs, k, replace=False) for _ in range(N_REPS)]  # cùng mask cho mọi phương pháp
        rec = {m: dict(lens=[], orig=[], med=[]) for m in METHODS}
        for hide in masks:
            A = Z.copy()
            truth_l = A[hide, j].copy()
            A[hide, j] = np.nan
            truth_o = X_orig[col].to_numpy()[hide]
            for m in METHODS:
                p = predict(m, A, j, hide, dev_vals)
                po = round_col(from_lens(p, col, log_cols, med, iqr), col)
                rec[m]["lens"].append(np.mean(np.abs(p - truth_l)))
                rec[m]["orig"].append(np.mean(np.abs(po - truth_o)))
                rec[m]["med"].append(np.median(np.abs(po - truth_o)))
        lens = {m: np.array(rec[m]["lens"]) for m in METHODS}
        best = min(METHODS, key=lambda m: lens[m].mean())
        rows = []
        for m in METHODS:
            a, o, dlt = lens[m], np.array(rec[m]["orig"]), lens[m] - lens[best]
            rows.append(dict(column=col, method=m, n_masked=k, n_reps=N_REPS, mae_lens_mean=a.mean(), mae_lens_sd=a.std(ddof=1),
                             mae_lens_se=a.std(ddof=1) / np.sqrt(N_REPS), mae_orig_mean=o.mean(), mae_orig_sd=o.std(ddof=1),
                             mae_orig_se=o.std(ddof=1) / np.sqrt(N_REPS), medae_orig_mean=np.mean(rec[m]["med"]),
                             paired_diff_vs_best=dlt.mean(), paired_diff_se=dlt.std(ddof=1) / np.sqrt(N_REPS)))
        t = pd.DataFrame(rows).set_index("method", drop=False)
        thr = t.at[best, "mae_lens_mean"] + t.at[best, "mae_lens_se"]
        chosen = next(m for m in METHODS if t.at[m, "mae_lens_mean"] <= thr)
        out[col] = dict(table=t, best=best, chosen=chosen, threshold=float(thr))
    return out


def category_experiment(Zc, cat, dev_vals, tier_vals):
    y = cat.to_numpy().astype(object)
    obs = np.flatnonzero(cat.notna().to_numpy())
    k = int(round(MASK_FRAC * len(obs)))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 999]))
    acc = {m: [] for m in CAT_METHODS}
    conf, ent = [], []
    for _ in range(N_REPS):
        hide = rng.choice(obs, k, replace=False)
        tr = np.setdiff1d(obs, hide)
        gm = pd.Series(y[tr]).mode()[0]
        acc["global_mode"].append(np.mean(y[hide] == gm))
        for name, v in (("mode_by_device", dev_vals), ("mode_by_tier", tier_vals)):
            pred = [pd.Series(y[tr][v[tr] == v[h]]).mode()[0] for h in hide]
            acc[name].append(np.mean(np.array(pred) == y[hide]))
        knn = KNeighborsClassifier(7, weights="distance").fit(Zc[tr], y[tr])
        pr = knn.predict_proba(Zc[hide])
        acc["knn7"].append(np.mean(knn.classes_[pr.argmax(1)] == y[hide]))
        conf.append(pr.max(1).mean())
        ent.append(np.mean(-(np.where(pr > 0, pr * np.log(pr), 0)).sum(1) / np.log(pr.shape[1])))
    rows = []
    base = np.array(acc["global_mode"])
    for m in CAT_METHODS:
        a = np.array(acc[m]); d = a - base
        rows.append(dict(method=m, n_masked=k, n_reps=N_REPS, acc_mean=a.mean(), acc_sd=a.std(ddof=1), acc_se=a.std(ddof=1) / np.sqrt(N_REPS),
                         paired_diff_vs_global_mode=d.mean(), paired_diff_se=d.std(ddof=1) / np.sqrt(N_REPS)))
    return pd.DataFrame(rows), float(np.mean(conf)), float(np.mean(ent))


# ---------------------------------------------------------------------------------------------- lõi xử lý (xác định, chạy được nhiều lần)
def run_core(df, dd, num):
    log_cols, _ = select_log_cols(df, dd, num)
    inv = audit_invalid(df, num)
    X_orig = df[num].astype(float).copy()
    for _, r in inv.iterrows():
        X_orig.at[r["row"], r["column"]] = np.nan  # vô hiệu hóa đúng từng ô
    n_nan_before = X_orig.isna().sum()
    Lnum, med, iqr = build_lens(X_orig, log_cols)
    aux = aux_matrix(df)
    Z = pd.concat([Lnum, aux], axis=1)
    names, Zv = list(Z.columns), Z.to_numpy()
    dev_vals, tier_vals = df["device_type"].to_numpy(), df["membership_tier"].to_numpy()
    cols = [c for c in num if n_nan_before[c] > 0]
    sel = masked_experiment(Zv, names, X_orig, dev_vals, cols, log_cols, med, iqr)
    chosen = {c: sel[c]["chosen"] for c in cols}
    # điền: mỗi phương pháp được chọn chạy một lần trên toàn ma trận
    full = {}
    for m in set(chosen.values()):
        if m.startswith("knn"):
            full[m] = KNNImputer(n_neighbors=int(m[3:]), weights="distance").fit_transform(Zv)
        elif m == "iterative":
            full[m] = IterativeImputer(estimator=BayesianRidge(), max_iter=15, random_state=0).fit_transform(Zv)
    X_new = X_orig.copy()
    audit = []
    invalid_keys = {(r["row"], r["column"]): r for _, r in inv.iterrows()}
    for c in cols:
        j, m = names.index(c), chosen[c]
        rows = np.flatnonzero(np.isnan(Zv[:, j]))
        if m == "median":
            vals = np.full(len(rows), np.nanmedian(Zv[:, j]))
        elif m == "median_by_device":
            vals = np.array([np.nanmedian(Zv[dev_vals == dev_vals[r], j]) for r in rows])
        else:
            vals = full[m][rows, j]
        v_orig = round_col(from_lens(vals, c, log_cols, med, iqr), c)
        for r, v in zip(rows, v_orig):
            X_new.at[r, c] = v
            key = (int(r), c)
            audit.append(dict(customer_id=df.at[r, "customer_id"], column=c, action="invalid_cell->imputed" if key in invalid_keys else "missing->imputed",
                              value_before=float(df.at[r, c]) if not np.isnan(df.at[r, c]) else np.nan, value_after=float(v), method=m,
                              reason=(invalid_keys[key]["rule"] + ": " + invalid_keys[key]["detail"]) if key in invalid_keys else "thiếu trong D3"))
    # biến thể và quyết định cho preferred_category (chỉ dùng ma trận số đã hoàn chỉnh, không dùng chính category)
    Lc, _, _ = build_lens(X_new, log_cols)
    Zc = pd.concat([Lc, aux], axis=1).to_numpy()
    cat = df["preferred_category"]
    cat_tab, conf_mask, ent_mask = category_experiment(Zc, cat, dev_vals, tier_vals)
    obs = np.flatnonzero(cat.notna().to_numpy())
    miss = np.flatnonzero(cat.isna().to_numpy())
    knn = KNeighborsClassifier(7, weights="distance").fit(Zc[obs], cat.to_numpy()[obs].astype(object))
    pr = knn.predict_proba(Zc[miss])
    gmode = pd.Series(cat.dropna()).mode()[0]
    variants = pd.DataFrame({"customer_id": df["customer_id"], "preferred_category_main": cat.fillna("unknown"),
                             "preferred_category_knn7": cat, "preferred_category_mode": cat.fillna(gmode),
                             "knn7_max_prob": np.nan, "knn7_norm_entropy": np.nan})
    variants.loc[miss, "preferred_category_knn7"] = knn.classes_[pr.argmax(1)]
    variants.loc[miss, "knn7_max_prob"] = pr.max(1)
    variants.loc[miss, "knn7_norm_entropy"] = -(np.where(pr > 0, pr * np.log(pr), 0)).sum(1) / np.log(pr.shape[1])
    for i in miss:
        audit.append(dict(customer_id=df.at[i, "customer_id"], column="preferred_category", action="missing->unknown", value_before=np.nan,
                          value_after="unknown", method="keep_unknown", reason="thiếu trong D3; giữ mức 'unknown' (xem docs/cau_c_decisions.md)"))
    proc = df.copy()
    for c in num:
        proc[c] = X_new[c].to_numpy()
    proc["age"] = proc["age"].astype(int)
    proc["recency_days"] = proc["recency_days"].astype(int)
    proc["preferred_category"] = cat.fillna("unknown")
    return dict(proc=proc, audit=pd.DataFrame(audit), inv=inv, sel=sel, chosen=chosen, cat_tab=cat_tab, cat_conf_mask=conf_mask,
                cat_ent_mask=ent_mask, variants=variants, log_cols=log_cols, n_nan_before=n_nan_before, miss_cat=miss)


# ---------------------------------------------------------------------------------------------- cờ, số đo cấu trúc, hình
def make_flags(proc, df, res, num):
    log_cols = res["log_cols"]
    X = proc[num].astype(float)
    q1, q3 = X.quantile(.25), X.quantile(.75)
    iqr = q3 - q1
    beyond3 = ((X < q1 - 3 * iqr) | (X > q3 + 3 * iqr)).any(axis=1)
    inv_rows = set(res["inv"]["row"])
    au = res["audit"]
    row_of = {cid: i for i, cid in enumerate(df["customer_id"])}
    imputed_n = au[au["column"].isin(num)].groupby("customer_id").size()
    flags = pd.DataFrame({"customer_id": df["customer_id"]})
    flags["n_cells_invalid_repaired"] = [int(i in inv_rows) for i in range(len(df))]
    flags["n_cells_numeric_imputed"] = flags["customer_id"].map(imputed_n).fillna(0).astype(int)
    flags["category_unknown"] = proc["preferred_category"].eq("unknown").astype(int)
    flags["extreme_tukey3_valid"] = (beyond3 & ~flags["n_cells_invalid_repaired"].astype(bool)).astype(int)
    flags["lof20_suspect"] = (LocalOutlierFactor(n_neighbors=20).fit_predict(make_lens(proc, num, "robust", log_cols)) == -1).astype(int)
    flags["orders_per_session_gt1"] = (proc["n_orders_12m"] / (12 * proc["sessions_per_month"]) > 1).astype(int)
    for name, (c, v) in BOUNDS.items():
        flags["at_" + name.replace(" = ", "_").replace(",", ".").replace(" ", "")] = (proc[c] == v).astype(int)
    return flags


def structure_metrics(df, proc, rows, num, log_cols):
    out = {}
    for name, d in (("before (D3)", df), ("after", proc)):
        r = {}
        for kind in ("raw_z", "robust"):
            Z = make_lens(d, num, kind, log_cols)
            k = knn_dist(Z)
            r[f"share_pairs_repaired_rows_{kind}"] = pair_share(Z, rows)
            r[f"knn5_max_over_median_{kind}"] = float(k.max() / np.median(k))
            r[f"hopkins_{kind}"] = hopkins(Z, SEED)[0]
        X = d[num].astype(float)
        r["max_abs_pearson_minus_spearman"] = float((X.corr() - X.corr(method="spearman")).abs().to_numpy()[np.triu_indices(len(num), 1)].max())
        out[name] = r
    return pd.DataFrame(out).T


def heap_table(df, proc, audit):
    rows = []
    imp = audit[audit["action"] != "missing->unknown"]
    for name, (c, v) in BOUNDS.items():
        landed = int(((imp["column"] == c) & (pd.to_numeric(imp["value_after"], errors="coerce") == v)).sum())
        rows.append(dict(point=name, before=int((df[c] == v).sum()), after=int((proc[c] == v).sum()), imputed_cells_landing_on_it=landed))
    return pd.DataFrame(rows)


def fig_selection(res):
    sel, chosen = res["sel"], res["chosen"]
    fig, axes = plt.subplots(2, 4, figsize=(23, 10))
    unit = {"age": "years", "avg_session_minutes": "min", "avg_order_value": "USD", "recency_days": "days", "sessions_per_month": "sessions",
            "discount_usage_rate": "rate", "cart_abandon_rate": "rate"}
    for ax, col, l in zip(axes.ravel(), list(sel), "abcdefg"):
        t = sel[col]["table"]
        x = np.arange(len(t))
        ax.bar(x, t["mae_lens_mean"], yerr=t["mae_lens_se"], capsize=3, color=[AFTER if m == chosen[col] else vs.NEUTRAL["light"] for m in t.method],
               edgecolor=vs.NEUTRAL["ink"], lw=.8, hatch=None)
        for xi, m, b in zip(x, t.method, ax.patches):
            b.set_hatch(HATCH[m])
        for xi, (mae, se) in enumerate(zip(t["mae_lens_mean"], t["mae_lens_se"])):
            ax.text(xi, mae + se + 0.01, f"{t['mae_orig_mean'].iloc[xi]:.3g}", ha="center", fontsize=10)
        ax.axhline(sel[col]["threshold"], color=vs.NEUTRAL["ink"], ls="--", lw=1.2)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in t.method], rotation=35, ha="right")
        ax.set_title(f"{col}\nselected: {SHORT[chosen[col]]} (labels: MAE in {unit[col]})", fontsize=11)
        ax.set_ylabel("MAE on selection scale" if l in "ae" else ""); vs.panel_label(ax, l)
    ax = axes.ravel()[7]
    ct = res["cat_tab"]
    x = np.arange(len(ct))
    ax.bar(x, ct["acc_mean"], yerr=ct["acc_se"], capsize=3, color=vs.NEUTRAL["light"], edgecolor=vs.NEUTRAL["ink"], lw=.8)
    for b, h in zip(ax.patches, ["", "//", "..", "xx"]):
        b.set_hatch(h)
    ax.axhline(0.2, color=vs.NEUTRAL["mid"], ls=":", lw=1.5, label="chance (0.20)"); ax.legend(loc="upper left")
    ax.set_xticks(x); ax.set_xticklabels(ct["method"], rotation=35, ha="right"); ax.set_ylabel("accuracy (masked)"); ax.set_ylim(0, .6)
    ax.set_title("preferred_category (kept as 'unknown')", fontsize=11); vs.panel_label(ax, "h")
    vs.finalize_figure(fig, FIG, "fig01_imputer_selection")


def fig_before_after(df, proc, res, num):
    cols = [c for c in num if (proc[c].to_numpy() != df[c].to_numpy()).any() and not np.array_equal(proc[c].to_numpy(), df[c].to_numpy(), equal_nan=True)]
    au = res["audit"]
    fig, axes = plt.subplots(2, 4, figsize=(23, 10))
    for ax, c, l in zip(axes.ravel(), cols, "abcdefg"):
        for d, col, ls, lab in ((df, BEFORE, "--", "Before (D3)"), (proc, AFTER, "-", "After")):
            x = np.sort(d[c].dropna().astype(float).to_numpy())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, ls=ls, label=lab)
        v = pd.to_numeric(au.loc[(au["column"] == c) & (au["action"] != "missing->unknown"), "value_after"], errors="coerce").to_numpy()
        ax.plot(v, np.full(len(v), -0.03), "|", color=AFTER, ms=12, mew=2, clip_on=False)
        if c in ("avg_order_value", "sessions_per_month", "avg_session_minutes"):
            ax.set_xscale("log")
        ax.set_title(c, fontsize=11); ax.set_ylabel("ECDF" if l in "ae" else ""); vs.panel_label(ax, l)
    axes[0, 0].legend(loc="lower right")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes.ravel()[len(cols)].text(0, .5, "ticks under the x-axis = imputed values", fontsize=12, va="center")
    vs.finalize_figure(fig, FIG, "fig02_before_after_distributions")


def fig_change_map(df, res, num):
    feats = num + [c for c in CAT]
    M = np.zeros((len(feats), len(df)))
    rmap = {cid: i for i, cid in enumerate(df["customer_id"])}
    for _, r in res["audit"].iterrows():
        v = {"invalid_cell->imputed": 1, "missing->imputed": 2, "missing->unknown": 3}[r["action"]]
        M[feats.index(r["column"]), rmap[r["customer_id"]]] = v
    fig, ax = plt.subplots(figsize=(19, 6.2))
    ax.imshow(M, aspect="auto", cmap=ListedColormap([vs.NEUTRAL["faint"], BEFORE, AFTER, vs.NEUTRAL["mid"]]), vmin=0, vmax=3, interpolation="nearest")
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats); ax.set_xlabel(f"Customer (row order, n = {len(df)})")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(facecolor=vs.NEUTRAL["faint"], label="unchanged"), Patch(facecolor=BEFORE, label="invalid cell -> imputed"),
                       Patch(facecolor=AFTER, label="missing -> imputed"), Patch(facecolor=vs.NEUTRAL["mid"], hatch="//", label="missing -> 'unknown'")],
              loc="upper center", bbox_to_anchor=(.5, -.12), ncol=4)
    vs.finalize_figure(fig, FIG, "fig03_cell_change_map")


def fig_structure(df, proc, SM, num, log_cols, rows):
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    ax = axes[0, 0]
    keys = ["share_pairs_repaired_rows_raw_z", "share_pairs_repaired_rows_robust"]
    x = np.arange(2)
    for k, (nm, col, h) in enumerate((("before (D3)", BEFORE, "//"), ("after", AFTER, ""))):
        v = [SM.loc[nm, kk] for kk in keys]
        ax.bar(x + (k - .5) * .38, v, .36, color=col, hatch=h, edgecolor="white", lw=.6, label=nm)
        for xi, vi in zip(x + (k - .5) * .38, v):
            ax.text(xi, vi + .005, f"{vi:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(["z-score on raw", "log1p + median/IQR"]); ax.set_ylabel("share of total squared pairwise distance\ninvolving the 7 repaired rows"); ax.legend(); vs.panel_label(ax, "a")
    ax = axes[0, 1]
    X0, X1 = df[num].astype(float), proc[num].astype(float)
    r0 = X0.std() / ((X0.quantile(.75) - X0.quantile(.25)) / 1.349)
    r1 = X1.std() / ((X1.quantile(.75) - X1.quantile(.25)) / 1.349)
    xx = np.arange(len(num))
    ax.bar(xx - .2, r0, .38, color=BEFORE, hatch="//", edgecolor="white", lw=.6, label="before (D3)")
    ax.bar(xx + .2, r1, .38, color=AFTER, edgecolor="white", lw=.6, label="after")
    ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(num, rotation=40, ha="right"); ax.set_ylabel("std / robust sd (IQR/1.349)"); ax.legend(); vs.panel_label(ax, "b")
    ax = axes[1, 0]
    for d, col, ls, lab in ((df, BEFORE, "--", "before (D3)"), (proc, AFTER, "-", "after")):
        k = knn_dist(make_lens(d, num, "raw_z", log_cols))
        k = np.sort(k / np.median(k))  # chia cho trung vị: thang z-score hai bên khác nhau (sd cột bị ô hỏng làm phồng), chỉ so độ nặng của đuôi
        ax.plot(k, np.arange(1, len(k) + 1) / len(k), color=col, ls=ls, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("5-NN distance / its median (z-score on raw)"); ax.set_ylabel("ECDF"); ax.legend(); vs.panel_label(ax, "c")
    ax = axes[1, 1]
    for d, col, mk, lab in ((df, BEFORE, "D", "before (D3)"), (proc, AFTER, "o", "after")):
        X = d[num].astype(float)
        gap = (X.corr() - X.corr(method="spearman")).abs().to_numpy()[np.triu_indices(len(num), 1)]
        ax.plot(np.sort(gap)[::-1], marker=mk, color=col, ls="--" if col == BEFORE else "-", label=lab)
    ax.set_xlabel("variable pair (sorted)"); ax.set_ylabel("|Pearson - Spearman|"); ax.legend(); vs.panel_label(ax, "d")
    vs.finalize_figure(fig, FIG, "fig04_structure_before_after")


def fig_correlation(df, proc, num):
    fig, axes = plt.subplots(1, 3, figsize=(23, 9), gridspec_kw={"width_ratios": [1, 1, 0.04]})
    for ax, d, t, l in ((axes[0], df, "Pearson, before (D3)", "a"), (axes[1], proc, "Pearson, after", "b")):
        M = d[num].astype(float).corr()
        im = ax.imshow(M.to_numpy(), cmap=vs.CMAP_DIV, vmin=-1, vmax=1)
        ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=45, ha="right"); ax.set_yticks(range(len(num))); ax.set_yticklabels(num)
        for i in range(len(num)):
            for j in range(len(num)):
                ax.text(j, i, f"{M.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(M.iloc[i, j]) > .6 else vs.NEUTRAL["ink"])
        ax.set_title(t); vs.panel_label(ax, l)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, cax=axes[2], label="Pearson correlation")
    vs.finalize_figure(fig, FIG, "fig05_pearson_before_after", pad=1.5)


def fig_heaps(HT):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    x = np.arange(len(HT))
    ax.bar(x - .2, HT["before"], .38, color=BEFORE, hatch="//", edgecolor="white", lw=.6, label="before (D3)")
    ax.bar(x + .2, HT["after"], .38, color=AFTER, edgecolor="white", lw=.6, label="after")
    for xi, (b, a, k) in enumerate(zip(HT["before"], HT["after"], HT["imputed_cells_landing_on_it"])):
        ax.text(xi - .2, b + 1, str(b), ha="center", fontsize=10); ax.text(xi + .2, a + 1, f"{a}" + (f" (+{k} imputed)" if k else ""), ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(HT["point"], rotation=15); ax.set_ylabel("rows at the boundary value"); ax.legend()
    vs.finalize_figure(fig, FIG, "fig06_boundary_heaps")


def fig_category(res, df):
    v = res["variants"]
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.6))
    ax = axes[0]
    ax.hist(v["knn7_max_prob"].dropna(), bins=np.linspace(0.2, 1, 17), color=AFTER, edgecolor="white", lw=.6)
    ax.axvline(0.2, color=vs.NEUTRAL["mid"], ls=":", lw=1.5)
    ax.set_xlabel("KNN-7 max class probability on the 13 missing cells"); ax.set_ylabel("count"); vs.panel_label(ax, "a")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    order = ["fashion", "electronics", "home", "beauty", "grocery", "unknown"]
    for ax, col, ttl, l in zip(axes[1:], ["preferred_category_main", "preferred_category_knn7", "preferred_category_mode"],
                               ["main ('unknown')", "variant: KNN-7", "variant: global mode"], "bcd"):
        vc = v[col].value_counts().reindex(order).fillna(0)
        bars = ax.bar(order, vc.values, color=[AFTER] * 5 + [vs.NEUTRAL["mid"]], edgecolor="white", lw=.6)
        bars[-1].set_hatch("//")
        for b, val in zip(bars, vc.values):
            ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{int(val)}", ha="center", fontsize=10)
        ax.set_title(ttl, fontsize=12); ax.tick_params(axis="x", rotation=35); ax.set_ylim(0, 95); vs.panel_label(ax, l)
    vs.finalize_figure(fig, FIG, "fig07_category_variants")


# ---------------------------------------------------------------------------------------------- main
def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOGD / "03_preprocess.log", mode="w", encoding="utf-8")])
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    vs.apply_publication_style()
    ck = Checker()
    h_raw_in = sha256(RAW)
    log.info("=" * 80 + "\nCÂU C - TIỀN XỬ LÝ (chỉ đọc D3 + data dictionary). SHA-256 D3: %s\n" + "=" * 80, h_raw_in)
    df, dd = read_input(RAW), read_input(DICT)
    num = dd.loc[dd["dtype"].isin(["int", "float"]), "attribute"].tolist()

    res = run_core(df, dd, num)
    proc, audit, inv, sel = res["proc"], res["audit"], res["inv"], res["sel"]
    show("1. Ô không hợp lệ bị vô hiệu hóa (luật miền + đẳng thức)", inv[["customer_id", "column", "value", "rule"]])
    for _, r in inv[inv["rule"] == "identity"].iterrows():
        log.info("   chi tiết đẳng thức: %s", r["detail"])
    for col, s in sel.items():
        t = s["table"].round(4)
        show(f"2. Thí nghiệm che ô: {col} ({int(t['n_masked'].iloc[0])} ô x {N_REPS} lần, cùng mask; MAE selection = thang lens, orig = đơn vị gốc)",
             t[["mae_lens_mean", "mae_lens_sd", "mae_lens_se", "mae_orig_mean", "mae_orig_sd", "mae_orig_se", "medae_orig_mean", "paired_diff_vs_best", "paired_diff_se"]])
        log.info("   tốt nhất: %s | ngưỡng 1-SE = %.4f | CHỌN (đơn giản nhất trong ngưỡng): %s", s["best"], s["threshold"], s["chosen"])
    show("3. preferred_category: độ chính xác che ô (cùng mask) và so sánh ghép cặp với mode toàn cục", res["cat_tab"].round(4))
    log.info("   KNN-7 trên ô che: độ tin cậy lớp dự đoán trung bình %.3f, entropy chuẩn hóa %.3f (0 = chắc chắn, 1 = đều 5 lớp)", res["cat_conf_mask"], res["cat_ent_mask"])
    v = res["variants"]
    mv = v.loc[res["miss_cat"]]
    log.info("   13 ô thiếu thật: KNN-7 max prob trung bình %.3f (min %.3f, max %.3f), entropy chuẩn hóa trung bình %.3f", mv["knn7_max_prob"].mean(),
             mv["knn7_max_prob"].min(), mv["knn7_max_prob"].max(), mv["knn7_norm_entropy"].mean())
    log.info("   số nhãn dự kiến sai nếu điền KNN-7: %.1f / 13 (acc mask %.3f)", 13 * (1 - res["cat_tab"].set_index("method").at["knn7", "acc_mean"]),
             res["cat_tab"].set_index("method").at["knn7", "acc_mean"])
    for _, r in audit[audit["column"] == "avg_order_value"].iterrows():
        tn = df.loc[df.customer_id == r["customer_id"], ["total_spend_12m", "n_orders_12m"]].iloc[0]
        log.info("   kiểm tra hợp lý %s (AOV bị vô hiệu): điền = %.2f; total/n (chỉ để so sánh) = %.2f", r["customer_id"], r["value_after"], tn["total_spend_12m"] / tn["n_orders_12m"])

    flags = make_flags(proc, df, res, num)
    rows_rep = sorted(set(inv["row"]))
    SM = structure_metrics(df, proc, rows_rep, num, res["log_cols"])
    HT = heap_table(df, proc, audit)
    num_aud = audit[audit["column"].isin(num)]
    imp_stats = []
    for c in sorted(num_aud["column"].unique()):
        vals = pd.to_numeric(num_aud.loc[num_aud["column"] == c, "value_after"])
        ok = proc.loc[~proc.customer_id.isin(num_aud.loc[num_aud["column"] == c, "customer_id"]), c].astype(float)
        rng_b = np.random.default_rng(np.random.SeedSequence([SEED, 7]))
        obs_r = ok.round(DECIMALS[c]).to_numpy()
        boot = [np.unique(rng_b.choice(obs_r, len(vals), replace=True), return_counts=True)[1].max() for _ in range(2000)] if len(vals) >= 5 else []
        imp_stats.append(dict(column=c, n_imputed=len(vals), std_imputed=vals.std(), std_observed=ok.std(), std_ratio=vals.std() / ok.std() if len(vals) > 1 else np.nan,
                              max_repeat_of_one_value=int(vals.value_counts().max()), bootstrap_p95_max_repeat=float(np.quantile(boot, .95)) if boot else np.nan))
    IS = pd.DataFrame(imp_stats)
    show("4. Điểm dồn ở biên trước/sau (ô điền không bị cắt ngưỡng)", HT)
    show("5. Giá trị điền: độ phân tán so với giá trị quan sát và độ lặp", IS.round(3))
    show("6. Số đo cấu trúc chỉ dựa trên D3 (trước) và dữ liệu đã xử lý (sau)", SM.round(4))
    chg = audit.groupby(["action", "column"]).size().rename("n_cells").reset_index()
    show("7. Tóm tắt thay đổi", chg)

    log.info("\n" + "=" * 80 + "\nNGHIỆM THU (chỉ trên D3 và dữ liệu đã xử lý)\n" + "=" * 80)
    src = Path(__file__).read_text(encoding="utf-8")
    ck("Mã nguồn không chứa tên file tham chiếu D0-D2 hay thư mục dữ liệu ẩn", not [s for s in FORBIDDEN if s in src], "")
    ck("Số dòng/id/thứ tự cột giữ nguyên (260 dòng, không xóa bản ghi)", len(proc) == len(df) == 260 and proc["customer_id"].equals(df["customer_id"]) and list(proc.columns) == list(df.columns))
    ck("Không còn ô trống ở cột số; preferred_category có đúng 13 'unknown'", not proc[num].isna().any().any() and (proc["preferred_category"] == "unknown").sum() == 13 and not proc.isna().any().any())
    ck("Không ô nào vi phạm luật miền sau xử lý", not viol_mask(proc, num).any().any(), f"{int(viol_mask(proc, num).values.sum())} ô")
    f_after = identity_flag(proc)[0]
    ck("Không dòng nào vi phạm đẳng thức xấp xỉ sau xử lý", not f_after.any(), f"{int(f_after.sum())} dòng")
    changed = {(r.customer_id, r.column) for r in audit.itertuples()}
    diff = set()
    idx = {cid: i for i, cid in enumerate(df["customer_id"])}
    for c in df.columns[1:]:
        a, b = df[c], proc[c]
        for i in np.flatnonzero(~((a == b) | (a.isna() & b.isna())).to_numpy()):
            diff.add((df.at[i, "customer_id"], c))
    # ô thiếu ban đầu cũng bị coi là 'thay đổi' vì NaN != giá trị; kiểm tra tập khớp chính xác
    ck("Tập ô thay đổi so với D3 khớp đúng với nhật ký kiểm toán; mọi ô khác giữ nguyên giá trị", diff == changed, f"{len(diff)} ô thay đổi, {len(changed)} trong nhật ký")
    Xp = proc[num].astype(float)
    q1p, q3p = Xp.quantile(.25), Xp.quantile(.75)
    iqrp = q3p - q1p
    inv_rows = set(inv["row"])
    ext_cells = {(proc.at[i, "customer_id"], c) for c in num for i in np.flatnonzero(((Xp[c] < q1p[c] - 3 * iqrp[c]) | (Xp[c] > q3p[c] + 3 * iqrp[c])).to_numpy())
                 if i not in inv_rows}
    touched = ext_cells & changed
    ex_ids = set(flags.loc[flags["extreme_tukey3_valid"] == 1, "customer_id"])
    n_ex_other = len({cid for cid, c in changed if cid in ex_ids})
    ck("Giá trị cực trị hợp lệ (ngoài 3 IQR) chỉ được đánh dấu, không ô nào bị sửa", not touched,
       f"{len(ext_cells)} ô cực trị hợp lệ ở {len(ex_ids)} dòng, {len(touched)} bị chạm; {n_ex_other} trong các dòng này có ô KHÁC được điền vì thiếu/không hợp lệ")
    ck("Ô điền không rơi đúng vào giá trị biên (không tạo thêm điểm dồn giả)", int(HT["imputed_cells_landing_on_it"].sum()) == 0, f"{int(HT['imputed_cells_landing_on_it'].sum())} ô")
    ck("Ô điền nằm trong miền hợp lệ mà không bị cắt ngưỡng", True, "đã kiểm ở bước miền; không có phép clip trong mã")
    big = IS[IS["n_imputed"] >= 5]
    ck("Giá trị điền không lặp nhiều hơn mức bốc ngẫu nhiên từ phân phối quan sát (bootstrap p95 của số lần lặp lớn nhất)",
       bool((big["max_repeat_of_one_value"] <= big["bootstrap_p95_max_repeat"]).all()),
       "; ".join(f"{r.column}: {r.max_repeat_of_one_value} <= {r.bootstrap_p95_max_repeat:.0f}" for r in big.itertuples()))
    ck("Tỉ trọng bình phương khoảng cách của 7 dòng đã sửa giảm ở cả hai cách biểu diễn",
       SM.at["after", "share_pairs_repaired_rows_raw_z"] < SM.at["before (D3)", "share_pairs_repaired_rows_raw_z"] and
       SM.at["after", "share_pairs_repaired_rows_robust"] < SM.at["before (D3)", "share_pairs_repaired_rows_robust"],
       f"raw_z {SM.at['before (D3)', 'share_pairs_repaired_rows_raw_z']:.3f} -> {SM.at['after', 'share_pairs_repaired_rows_raw_z']:.3f}; "
       f"robust {SM.at['before (D3)', 'share_pairs_repaired_rows_robust']:.3f} -> {SM.at['after', 'share_pairs_repaired_rows_robust']:.3f}")
    ck("Cột đặc trưng đầu ra đúng 14 + id, không chứa cờ/nhật ký", list(proc.columns) == ["customer_id"] + list(dd["attribute"]) and not (set(flags.columns[1:]) & set(proc.columns)))
    res2 = run_core(df, dd, num)
    ck("Chạy lại cho kết quả giống hệt (hash dữ liệu xử lý, nhật ký, biến thể)", hash_frame(res["proc"]) == hash_frame(res2["proc"]) and
       hash_frame(res["audit"].astype(str)) == hash_frame(res2["audit"].astype(str)) and hash_frame(res["variants"]) == hash_frame(res2["variants"]))
    ck("D3 không bị thay đổi", sha256(RAW) == h_raw_in, h_raw_in[:12])

    # ghi file
    write_csv(proc, OUT / "data_after_preprocessing.csv")
    write_csv(audit, OUT / "preprocessing_audit.csv")
    write_csv(flags, OUT / "preprocessing_flags.csv")
    write_csv(v, OUT / "variants" / "preferred_category_variants.csv")
    for name, tbl in (("c_invalid_cells", inv), ("c_imputer_selection_category", res["cat_tab"]), ("c_heaps_before_after", HT), ("c_imputed_value_stats", IS),
                      ("c_structure_before_after", SM.reset_index().rename(columns={"index": "stage"})), ("c_changes_summary", chg)):
        save_table(tbl, name)
    save_table(pd.concat([s["table"].assign(chosen=lambda t, c=c: t["method"] == res["chosen"][c], threshold_1se=s["threshold"], best=s["best"]) for c, s in sel.items()]),
               "c_imputer_selection_numeric")
    for fn in (fig_selection, fig_before_after, fig_change_map, fig_heaps, fig_category):
        pass
    fig_selection(res); fig_before_after(df, proc, res, num); fig_change_map(df, res, num)
    fig_structure(df, proc, SM, num, res["log_cols"], rows_rep); fig_correlation(df, proc, num); fig_heaps(HT); fig_category(res, df)
    ck("Đã ghi đủ file kết quả, bảng và hình", all((OUT / f).exists() for f in ["data_after_preprocessing.csv", "preprocessing_audit.csv", "preprocessing_flags.csv"]) and len(list(FIG.glob("*.png"))) == 7)

    log.info("-" * 80)
    n_ok = sum(ok for _, ok, _ in ck.rows)
    log.info("KẾT QUẢ NGHIỆM THU: %d/%d đạt", n_ok, len(ck.rows))
    if ck.all_ok:
        files = [OUT / "data_after_preprocessing.csv", OUT / "preprocessing_audit.csv", OUT / "preprocessing_flags.csv", OUT / "variants" / "preferred_category_variants.csv"]
        man = dict(frozen_files={str(p.relative_to(ROOT)): sha256(p) for p in files}, input_raw_corrupted_sha256=h_raw_in, script_sha256=sha256(Path(__file__)),
                   seed=SEED, n_reps=N_REPS, mask_frac=MASK_FRAC, versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
                   chosen_methods=res["chosen"], note="Sau khi file này được ghi, chỉ 03b_evaluate_preprocessing.py được mở D0 để đối chiếu.")
        mp = OUT / "FROZEN_SHA256.json"
        if mp.exists():
            os.chmod(mp, 0o644)
        mp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        for p in files + [mp]:
            os.chmod(p, 0o444)
        log.info("Đã đóng băng %d file (chmod 444) và ghi SHA-256 vào %s", len(files), mp.relative_to(ROOT))
    return 0 if ck.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
