#!/usr/bin/env python3
"""Câu b - Data Understanding.

PHẦN A (mù): chỉ đọc data_raw_corrupted.csv (D3) và data_dictionary.csv. Không đọc D0-D2, nhãn ẩn, nhật ký chèn lỗi.
PHẦN B (đối chiếu): sau khi Phần A đã ghi xong mọi con số, mới đọc D0, D1, D2 và data/hidden/ để đo tác động của
nhiễu, ngoại lai, thiếu và để kiểm tra các phát hiện mù có đúng không.

Mọi ngưỡng/phép biến đổi dùng trong Phần A đều suy ra từ D3 hoặc là luật miền nghiệp vụ nêu rõ ở RULES.
Đầu ra: figures/02_understanding/*.png|pdf, reports/tables/*.csv, reports/tables/b_key_numbers.json, logs/02_data_understanding.log
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_style as vs  # noqa: E402
from dm_utils import (RULES, bowley, hopkins, identity_dev, identity_flag, knn_dist, knn_sets, make_lens,  # noqa: E402,F401
                      pair_share, pca2, select_log_cols, viol_mask)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "01_generated"
HID = ROOT / "data" / "hidden"
FIG = ROOT / "figures" / "02_understanding"
TAB = ROOT / "reports" / "tables"
LOGD = ROOT / "logs"
SEED = 42
STAGES = ["D0 clean", "D1 +noise", "D2 +outliers", "D3 +missing"]
CAT = ["membership_tier", "device_type", "preferred_category", "app_installed"]
SEGMENTS = ["HighValue", "PromoHunter", "Infrequent", "Browser"]
log = logging.getLogger("b")
KEY: dict = {}


# ---------------------------------------------------------------------------------------------- tiện ích
def show(title, obj):
    log.info("\n--- %s ---\n%s", title, obj.to_string() if hasattr(obj, "to_string") else obj)


def save_table(df, name, index=True):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=index)


def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(ct)[0]
    return float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))


def perm_chi2_p(ind, g, n=10000, seed=SEED):
    """p-value hoán vị của thống kê chi2 (đáng tin khi số kỳ vọng < 5)."""
    d = pd.DataFrame({"i": ind.to_numpy(), "g": g.to_numpy()}).dropna(subset=["g"])
    codes, k = pd.factorize(d["g"])[0], d["g"].nunique()
    y = d["i"].to_numpy().astype(int)
    rng = np.random.default_rng(seed)

    def stat(v):
        obs = np.vstack([np.bincount(codes[v == a], minlength=k) for a in (0, 1)]).astype(float)
        exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
        return float(((obs - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())
    s0 = stat(y)
    return (sum(stat(rng.permutation(y)) >= s0 - 1e-12 for _ in range(n)) + 1) / (n + 1)


def kw_eps(x, g):
    d = pd.DataFrame({"x": x, "g": g}).dropna()
    groups = [v["x"].to_numpy() for _, v in d.groupby("g")]
    H, p = stats.kruskal(*groups)
    return (H - len(groups) + 1) / (len(d) - len(groups)), p


# ---------------------------------------------------------------------------------------------- PHẦN A
def a1_overview(df, dd, num):
    show("A1. Kích thước", f"{df.shape[0]} dòng x {df.shape[1]} cột (gồm customer_id)")
    sch = pd.DataFrame({"dtype_pandas": df.dtypes.astype(str), "n_unique": df.nunique(), "n_missing": df.isna().sum()})
    sch = sch.join(dd.set_index("attribute")[["group", "dtype", "valid_domain"]].rename(columns={"dtype": "dtype_khai_bao"}))
    show("A1. Lược đồ: kiểu đọc được vs kiểu khai báo", sch)
    save_table(sch, "a1_schema")
    KEY["n_rows"], KEY["n_cols_features"] = len(df), len(df.columns) - 1
    KEY["id_unique"] = bool(df["customer_id"].is_unique)
    dup = int(df.drop(columns="customer_id").duplicated().sum())
    KEY["duplicate_rows_excluding_id"] = dup
    log.info("customer_id duy nhất: %s | dòng trùng (bỏ id): %d", KEY["id_unique"], dup)


def a2_descriptives(df, dd, num):
    X = df[num].astype(float)
    d = X.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    d["missing"] = X.isna().sum()
    d["skew"] = X.skew()
    d["kurtosis"] = X.kurt()
    d["bowley_skew"] = [bowley(X[c].dropna()) for c in num]
    d["mean/median"] = d["mean"] / d["50%"]
    d["CV"] = d["std"] / d["mean"]
    d["robust_sd(IQR/1.349)"] = (d["75%"] - d["25%"]) / 1.349
    d["std/robust_sd"] = d["std"] / d["robust_sd(IQR/1.349)"]
    show("A2. Thống kê mô tả (thô, chưa xử lý gì)", d.round(3))
    save_table(d.round(4), "a2_numeric_describe")
    rows = []
    for c in CAT:
        vc = df[c].value_counts(dropna=False)
        p = vc / vc.sum()
        rows.append(dict(attribute=c, n_levels=int(df[c].nunique()), n_missing=int(df[c].isna().sum()),
                         mode=str(vc.index[0]), mode_share=round(float(p.iloc[0]), 3),
                         entropy_bits=round(float(-(p * np.log2(p)).sum()), 3), counts=json.dumps({str(k): int(v) for k, v in vc.items()})))
    cdf = pd.DataFrame(rows).set_index("attribute")
    show("A2. Biến phân loại", cdf)
    save_table(cdf, "a2_categorical_describe")
    log_cols, sk = select_log_cols(df, dd, num)
    d["skew_excl_violations"] = sk["skew_excl_violations"]
    d["bounded_[0,1]"] = sk["bounded_[0,1]"]
    show("A2. Quy tắc chọn cột lệch phải cần log: skew sau khi loại ô vi phạm luật miền > 1 và không phải biến tỉ lệ bị chặn [0,1]",
         d[["skew", "skew_excl_violations", "bowley_skew", "bounded_[0,1]"]].assign(chon_log=[c in log_cols for c in num]).round(3))
    KEY["skew_raw"] = d["skew"].round(2).to_dict()
    KEY["bowley_skew"] = d["bowley_skew"].round(3).to_dict()
    KEY["skew_excl_violations"] = d["skew_excl_violations"].round(2).to_dict()
    KEY["log_cols"] = log_cols
    KEY["std_over_robust_sd"] = d["std/robust_sd"].round(2).to_dict()
    return d, log_cols


def a3_domain(df, num):
    V = viol_mask(df, num)
    rows = []
    for c in num:
        for i in np.flatnonzero(V[c]):
            rows.append(dict(customer_id=df.at[i, "customer_id"], column=c, value=float(df.at[i, c]), rule=RULES[c][:2]))
    vl = pd.DataFrame(rows)
    show("A3. Giá trị vi phạm luật miền", vl if len(vl) else "không có")
    save_table(vl, "a3_domain_violations", index=False)
    for c, v in {"sessions_per_month": 999, "avg_session_minutes": 1440}.items():
        pass
    sentinel = {c: int((df[c] == v).sum()) for c, v in [("sessions_per_month", 999), ("age", 999), ("age", 0), ("recency_days", -1), ("recency_days", 999)] if (df[c] == v).any()}
    neg = {c: int((df[c] < 0).sum()) for c in num if (df[c] < 0).any()}
    log.info("Giá trị 'mã đặc biệt' (999/0/-1) tìm thấy: %s | cột có giá trị âm: %s", sentinel or "không", neg or "không")
    KEY["n_domain_violation_cells"] = int(V.values.sum())
    KEY["n_domain_violation_rows"] = int(V.any(axis=1).sum())
    KEY["domain_violations"] = {r["column"]: r["value"] for r in rows}
    return V


def a4_missing(df, num):
    cols = [c for c in df.columns if c != "customer_id"]
    M = df[cols].isna()
    ms = pd.DataFrame({"n_missing": M.sum(), "pct": (100 * M.mean()).round(2)})
    show("A4. Giá trị thiếu theo cột", ms[ms.n_missing > 0] if (ms.n_missing > 0).any() else ms)
    save_table(ms, "a4_missing_by_column")
    per_row = M.sum(axis=1)
    log.info("Số ô thiếu mỗi dòng: %s | dòng đủ dữ liệu: %d/%d", per_row.value_counts().sort_index().to_dict(), int((per_row == 0).sum()), len(df))
    miss_cols = ms.index[ms.n_missing > 0].tolist()
    both = {}
    for i, a in enumerate(miss_cols):
        for b in miss_cols[i + 1:]:
            obs, exp = int((M[a] & M[b]).sum()), M[a].mean() * M[b].mean() * len(df)
            both[f"{a} & {b}"] = dict(observed=obs, expected_if_independent=round(float(exp), 2))
    show("A4. Thiếu đồng thời (quan sát vs kỳ vọng nếu độc lập)", pd.DataFrame(both).T)
    rows = []
    for mc in miss_cols:
        ind = df[mc].isna()
        for oc in num + CAT:
            if oc == mc:
                continue
            if oc in num:
                a, b = df.loc[ind, oc].dropna(), df.loc[~ind, oc].dropna()
                U, p = stats.mannwhitneyu(a, b)
                rows.append(dict(missing_col=mc, other=oc, test="Mann-Whitney", p_asymptotic=p, p=p, effect=2 * U / (len(a) * len(b)) - 1, effect_name="rank-biserial"))
            else:
                ct = pd.crosstab(ind, df[oc])
                chi2, p, _, exp = stats.chi2_contingency(ct)
                pp = perm_chi2_p(ind, df[oc]) if exp.min() < 5 else np.nan
                rows.append(dict(missing_col=mc, other=oc, test="Chi2", p_asymptotic=p, p=(pp if exp.min() < 5 else p), p_perm=pp,
                                 effect=np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))), effect_name="Cramer V", min_expected=float(exp.min())))
    t = pd.DataFrame(rows)
    t["p_bonferroni"] = np.minimum(t["p"] * len(t), 1.0)
    t = t.sort_values("p")
    show(f"A4. Kiểm định cơ chế thiếu ({len(t)} phép thử; p = hoán vị khi số kỳ vọng < 5, ngược lại p tiệm cận; Bonferroni theo p này)", t.head(10)[["missing_col", "other", "test", "p_asymptotic", "p_perm", "p", "p_bonferroni", "effect", "min_expected"]].round(4))
    save_table(t.round(5), "a4_missing_mechanism_tests", index=False)
    dev = df.assign(_m=df["avg_session_minutes"].isna()).groupby("device_type")["_m"].agg(["sum", "count", "mean"])
    show("A4. Tỉ lệ thiếu avg_session_minutes theo device_type", dev.round(3))
    KEY["missing_pct"] = ms.loc[miss_cols, "pct"].to_dict()
    KEY["missing_rate_session_minutes_by_device"] = dev["mean"].round(3).to_dict()
    KEY["missing_tests_top"] = t.head(6)[["missing_col", "other", "test", "p_asymptotic", "p", "p_bonferroni", "effect", "min_expected"]].round(4).to_dict("records")
    return M, miss_cols, dev


def a5_relations(df, num):
    X = df[num].astype(float)
    sp, pe = X.corr(method="spearman"), X.corr(method="pearson")
    pairs = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            pairs.append(dict(a=a, b=b, spearman=sp.loc[a, b], pearson_raw=pe.loc[a, b]))
    pr = pd.DataFrame(pairs)
    pr["abs_spearman"] = pr.spearman.abs()
    pr = pr.sort_values("abs_spearman", ascending=False)
    show("A5. Các cặp thuộc tính |Spearman| lớn nhất", pr.head(10).round(3))
    show("A5. Các cặp mà Pearson thô và Spearman lệch nhau nhiều (|chênh| > 0,25)", pr.assign(diff=(pr.pearson_raw - pr.spearman)).query("abs(diff) > 0.25")[["a", "b", "spearman", "pearson_raw", "diff"]].round(3))
    save_table(pr.round(4), "a5_pair_correlations", index=False)
    comp = X.dropna()
    R = comp.rank().corr().to_numpy()
    vif = pd.Series(np.diag(np.linalg.inv(R)), index=num).sort_values(ascending=False)
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    show(f"A5. VIF trên hạng (Spearman), {len(comp)} dòng đủ dữ liệu", vif.round(2))
    log.info("Trị riêng ma trận tương quan Spearman: %s | số PC để giữ 80%%/90%% phương sai: %d/%d", ev.round(2).tolist(), int((cum < 0.8).sum() + 1), int((cum < 0.9).sum() + 1))
    cv = pd.DataFrame(index=CAT, columns=CAT, dtype=float)
    for a in CAT:
        for b in CAT:
            cv.loc[a, b] = 1.0 if a == b else cramers_v(df[a], df[b])
    show("A5. Cramér V giữa các biến phân loại", cv.round(3))
    eps = pd.DataFrame(index=CAT, columns=num, dtype=float)
    epsp = eps.copy()
    for c in CAT:
        for n in num:
            eps.loc[c, n], epsp.loc[c, n] = kw_eps(X[n], df[c])
    show("A5. Epsilon^2 Kruskal-Wallis: mức biến số khác nhau giữa các nhóm của biến phân loại", eps.round(3))
    save_table(eps.round(4), "a5_kruskal_epsilon2")
    KEY["top_pairs"] = pr.head(6)[["a", "b", "spearman", "pearson_raw"]].round(3).to_dict("records")
    KEY["vif_rank"] = vif.round(2).to_dict()
    KEY["pc_for_80_90"] = [int((cum < 0.8).sum() + 1), int((cum < 0.9).sum() + 1)]
    return sp, pe, cv, eps, vif


def a6_outliers(df, num, log_cols, V):
    X = df[num].astype(float)
    flags = {"domain_rules": V.any(axis=1)}
    q1, q3 = X.quantile(.25), X.quantile(.75)
    iqr = q3 - q1
    flags["tukey_1.5"] = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).any(axis=1)
    flags["tukey_3"] = ((X < q1 - 3 * iqr) | (X > q3 + 3 * iqr)).any(axis=1)
    L = X.copy()
    for c in log_cols:
        L[c] = np.log1p(L[c].where(L[c] > -1))
    mad = 1.4826 * (L - L.median()).abs().median()
    flags["robust_z_3.5"] = (((L - L.median()) / mad).abs() > 3.5).any(axis=1)
    flags["identity_mismatch"], dev, med, sd = identity_flag(df)
    lof = LocalOutlierFactor(n_neighbors=20).fit_predict(make_lens(df, num, "robust", log_cols))
    flags["LOF_20"] = pd.Series(lof == -1, index=df.index)
    F = pd.DataFrame(flags)
    cnt = F.sum()
    show("A6. Số dòng bị mỗi phương pháp gắn cờ", cnt.to_frame("n_rows"))
    show("A6. Giao giữa các phương pháp (số dòng cùng bị gắn cờ)", pd.DataFrame({a: {b: int((F[a] & F[b]).sum()) for b in F} for a in F}))
    per_feat = pd.DataFrame({"domain": V.sum(), "tukey_1.5": ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).sum(),
                             "tukey_3": ((X < q1 - 3 * iqr) | (X > q3 + 3 * iqr)).sum()})
    show("A6. Số ô vượt ngưỡng theo thuộc tính", per_feat)
    log.info("Độ lệch đẳng thức ln(total/(n x AOV)): trung vị=%.3f, sd bền=%.3f (ngưỡng 5 sd bền = ±%.3f); sd thường=%.3f", med, sd, 5 * sd, dev.std())
    hard = F["domain_rules"] | F["identity_mismatch"]
    save_table(F.assign(customer_id=df.customer_id).set_index("customer_id").loc[F.any(axis=1).values], "a6_flagged_rows")
    KEY["outlier_rows_by_method"] = cnt.to_dict()
    KEY["hard_error_rows"] = int(hard.sum())
    KEY["identity_sd_robust"] = round(float(sd), 4)
    KEY["identity_sd_std"] = round(float(dev.std()), 4)
    return F, hard, dev, (med, sd), per_feat


def a7_distance(df, num, log_cols, hard):
    out = {}
    rawz, rob = make_lens(df, num, "raw_z", log_cols), make_lens(df, num, "robust", log_cols)
    X = df[num].astype(float)
    Xi = X.fillna(X.median())
    var = Xi.var(ddof=0)
    share = (var / var.sum()).sort_values(ascending=False)
    show("A7. Tỉ trọng phương sai thô của từng thuộc tính (quyết định đóng góp vào khoảng cách Euclid khi KHÔNG chuẩn hóa)", (100 * share).round(3).to_frame("pct_of_total_variance"))
    rows = np.flatnonzero(hard.to_numpy())
    for name, Z in [("raw_z", rawz), ("robust_log", rob)]:
        out[name] = dict(share_pairs_with_hard_rows=pair_share(Z, rows), knn5_median=float(np.median(knn_dist(Z))),
                         knn5_p95=float(np.quantile(knn_dist(Z), .95)), knn5_max=float(knn_dist(Z).max()), hopkins=hopkins(Z, SEED))
    raw_all = Xi.to_numpy()
    out["raw_unscaled"] = dict(share_pairs_with_hard_rows=pair_share(raw_all, rows))
    tab = pd.DataFrame(out).T
    tab["hard_rows_share_of_rows"] = len(rows) / len(df)
    show(f"A7. Khoảng cách: tác động của {len(rows)} dòng lỗi cứng (vi phạm luật miền hoặc đẳng thức) theo cách biểu diễn", tab.round(4))
    save_table(tab.map(lambda v: v if not isinstance(v, tuple) else v[0]).round(5), "a7_distance_lens")
    KEY["variance_share_raw_pct"] = (100 * share).round(3).to_dict()
    KEY["distance_lens"] = {k: {kk: (round(vv[0], 4) if isinstance(vv, tuple) else round(vv, 4)) for kk, vv in v.items()} for k, v in out.items()}
    return share, rawz, rob, out


# ---- hình Phần A
def fig_missing(df, M, miss_cols, dev):
    cols = [c for c in df.columns if c != "customer_id"]
    fig = plt.figure(figsize=(19, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 0.9])
    ax = fig.add_subplot(gs[0])
    pct = (100 * M[cols].mean()).sort_values()
    ax.barh(pct.index, pct.values, color=[vs.PALETTE["red"] if v > 0 else vs.NEUTRAL["light"] for v in pct], edgecolor="white", lw=.6)
    for y, v in enumerate(pct.values):
        ax.text(v + .2, y, f"{v:.1f}%", va="center", fontsize=11)
    ax.set_xlabel("Missing (%)"); ax.set_xlim(0, pct.max() * 1.25)
    vs.panel_label(ax, "a")
    ax = fig.add_subplot(gs[1])
    ax.imshow(M[miss_cols].T.to_numpy(), aspect="auto", cmap=ListedColormap([vs.NEUTRAL["faint"], vs.PALETTE["red"]]), interpolation="nearest")
    ax.set_yticks(range(len(miss_cols))); ax.set_yticklabels(miss_cols)
    ax.set_xlabel(f"Customer (row order, n = {len(df)})")
    for s in ax.spines.values(): s.set_visible(False)
    vs.panel_label(ax, "b")
    ax = fig.add_subplot(gs[2])
    ax.bar(dev.index, 100 * dev["mean"], color=vs.PALETTE["blue"], edgecolor="white", lw=.6, hatch="//")
    for x, (s, n, m) in enumerate(zip(dev["sum"], dev["count"], dev["mean"])):
        ax.text(x, 100 * m + .5, f"{int(s)}/{int(n)}", ha="center", fontsize=11)
    ax.set_ylabel("avg_session_minutes missing (%)"); ax.set_ylim(0, 100 * dev["mean"].max() * 1.25)
    vs.panel_label(ax, "c")
    vs.finalize_figure(fig, FIG, "fig02_missing_values")


def fig_hist_box(df, num, log_cols):
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, c in zip(axes.ravel(), num):
        x = df[c].dropna().astype(float)
        ax.hist(x, bins=30, color=vs.PALETTE["blue"], edgecolor="white", lw=.4)
        ax.axvline(x.median(), color=vs.NEUTRAL["ink"], ls="--", lw=1.6)
        ax.axvline(x.mean(), color=vs.PALETTE["red"], ls="-", lw=1.6)
        ax.set_title(c, fontsize=12); ax.set_ylabel("count" if c in (num[0], num[5]) else "")
        ax.text(.97, .95, f"skew {x.skew():.1f}\nmax {x.max():,.4g}", transform=ax.transAxes, ha="right", va="top", fontsize=10)
    axes[0, 0].plot([], [], color=vs.NEUTRAL["ink"], ls="--", label="median"); axes[0, 0].plot([], [], color=vs.PALETTE["red"], label="mean")
    axes[0, 0].legend(loc="center right", fontsize=10)
    vs.finalize_figure(fig, FIG, "fig03_histograms_raw")

    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, c in zip(axes.ravel(), log_cols):
        x = np.log1p(df[c].dropna().astype(float).clip(lower=0))
        ax.hist(x, bins=30, color=vs.PALETTE["teal"], edgecolor="white", lw=.4)
        ax.set_title(f"log1p({c})", fontsize=11); ax.text(.97, .95, f"skew {x.skew():.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=10)
    for ax in axes.ravel()[len(log_cols):]:
        ax.axis("off")
    vs.finalize_figure(fig, FIG, "fig04_histograms_log1p")

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, c in zip(axes.ravel(), num):
        x = df[c].dropna().astype(float)
        bp = ax.boxplot(x, orientation="vertical", widths=.5, patch_artist=True, flierprops=dict(marker="o", markersize=4, markerfacecolor=vs.PALETTE["red"], markeredgecolor="none"),
                        medianprops=dict(color=vs.NEUTRAL["ink"], lw=2), boxprops=dict(facecolor=vs.BLUE_FAMILY["light"], edgecolor=vs.PALETTE["blue"], lw=1.6),
                        whiskerprops=dict(color=vs.PALETTE["blue"], lw=1.6), capprops=dict(color=vs.PALETTE["blue"], lw=1.6))
        q1, q3 = x.quantile(.25), x.quantile(.75)
        n_out = int(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum())
        ax.set_title(f"{c}\n{n_out} beyond 1.5 IQR", fontsize=12); ax.set_xticks([])
    vs.finalize_figure(fig, FIG, "fig05_boxplots_raw")


def fig_categorical(df, num):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    orders = {"membership_tier": ["bronze", "silver", "gold"], "device_type": None, "preferred_category": None, "app_installed": [0, 1]}
    for ax, c in zip(axes, CAT):
        vc = df[c].value_counts(dropna=False)
        order = orders[c] or [k for k in vc.index if not pd.isna(k)]
        labels = [str(o) for o in order] + (["missing"] if df[c].isna().any() else [])
        vals = [int(vc.get(o, 0)) for o in order] + ([int(df[c].isna().sum())] if df[c].isna().any() else [])
        cols = [vs.PALETTE["blue"]] * len(order) + ([vs.NEUTRAL["mid"]] if df[c].isna().any() else [])
        bars = ax.bar(labels, vals, color=cols, edgecolor="white", lw=.6)
        if df[c].isna().any(): bars[-1].set_hatch("//")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}\n({100 * v / len(df):.0f}%)", ha="center", fontsize=10)
        ax.set_title(c, fontsize=12); ax.set_ylim(0, max(vals) * 1.25); ax.tick_params(axis="x", rotation=20)
        vs.panel_label(ax, "abcd"[CAT.index(c)])
    axes[0].set_ylabel("count")
    vs.finalize_figure(fig, FIG, "fig06_countplots")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    spec = [("membership_tier", "total_spend_12m", ["bronze", "silver", "gold"]), ("device_type", "sessions_per_month", ["mobile", "desktop", "tablet"]),
            ("preferred_category", "avg_order_value", ["fashion", "electronics", "home", "beauty", "grocery"])]
    for ax, (g, n, order) in zip(axes, spec):
        data = [df.loc[df[g] == o, n].astype(float).clip(lower=1e-2).to_numpy() for o in order]
        ax.boxplot(data, patch_artist=True, widths=.55, flierprops=dict(marker="o", markersize=4, markerfacecolor=vs.PALETTE["red"], markeredgecolor="none"),
                   medianprops=dict(color=vs.NEUTRAL["ink"], lw=2), boxprops=dict(facecolor=vs.BLUE_FAMILY["light"], edgecolor=vs.PALETTE["blue"], lw=1.6),
                   whiskerprops=dict(color=vs.PALETTE["blue"], lw=1.6), capprops=dict(color=vs.PALETTE["blue"], lw=1.6))
        ax.set_xticklabels([f"{o}\n(n={len(d)})" for o, d in zip(order, data)]); ax.set_yscale("log"); ax.set_ylabel(n); ax.set_xlabel(g)
        vs.panel_label(ax, "abc"[spec.index((g, n, order))])
    vs.finalize_figure(fig, FIG, "fig07_categorical_vs_numeric")


def fig_correlation(sp, pe, cv, eps):
    fig, axes = plt.subplots(1, 3, figsize=(23, 9), gridspec_kw={"width_ratios": [1, 1, 0.04]})
    for ax, M, t, l in [(axes[0], sp, "Spearman (rank-based, robust)", "a"), (axes[1], pe, "Pearson on raw values", "b")]:
        im = ax.imshow(M.to_numpy(), cmap=vs.CMAP_DIV, vmin=-1, vmax=1)
        ax.set_xticks(range(len(M))); ax.set_xticklabels(M.columns, rotation=55, ha="right", fontsize=11); ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=11)
        for i in range(len(M)):
            for j in range(len(M)):
                v = M.iloc[i, j]
                if i != j and abs(v) >= 0.5:  # chỉ ghi số cho cặp tương quan mạnh để chữ đủ lớn và không chồng nhau
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=11, color="white" if abs(v) > .6 else vs.NEUTRAL["ink"])
        ax.set_title(t, fontsize=13); vs.panel_label(ax, l)
        for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im, cax=axes[2], label="correlation")
    vs.finalize_figure(fig, FIG, "fig08_correlation_numeric", pad=1.5)

    fig, axes = plt.subplots(1, 2, figsize=(21, 5.6), gridspec_kw={"width_ratios": [0.7, 1.6]})
    ax = axes[0]; im = ax.imshow(cv.to_numpy(), cmap=vs.CMAP_SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(cv.columns, rotation=30, ha="right"); ax.set_yticks(range(4)); ax.set_yticklabels(cv.index)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{cv.iloc[i, j]:.2f}", ha="center", va="center", fontsize=11, color="white" if cv.iloc[i, j] > .5 else vs.NEUTRAL["ink"])
    ax.set_title("Cramér V between categoricals", fontsize=13); vs.panel_label(ax, "a")
    for s in ax.spines.values(): s.set_visible(False)
    ax = axes[1]; im = ax.imshow(eps.to_numpy().astype(float), cmap=vs.CMAP_SEQ, vmin=0, vmax=max(.5, float(np.nanmax(eps.to_numpy().astype(float)))))
    ax.set_xticks(range(eps.shape[1])); ax.set_xticklabels(eps.columns, rotation=40, ha="right"); ax.set_yticks(range(4)); ax.set_yticklabels(eps.index)
    for i in range(eps.shape[0]):
        for j in range(eps.shape[1]): ax.text(j, i, f"{eps.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white" if eps.iloc[i, j] > .25 else vs.NEUTRAL["ink"])
    ax.set_title("Kruskal-Wallis epsilon² (numeric differences across category groups)", fontsize=13); vs.panel_label(ax, "b")
    for s in ax.spines.values(): s.set_visible(False)
    vs.finalize_figure(fig, FIG, "fig09_categorical_association", pad=1.5)


def fig_scatter(df, dev, thr):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    spec = [("n_orders_12m", "total_spend_12m", True), ("sessions_per_month", "pages_per_session", True), ("discount_usage_rate", "avg_order_value", True)]
    for ax, (a, b, lg), l in zip(axes.ravel(), spec, "abc"):
        ax.scatter(df[a], df[b], s=22, color=vs.PALETTE["blue"], alpha=.7, edgecolor="none")
        if lg:
            if a != "discount_usage_rate": ax.set_xscale("log")
            ax.set_yscale("log")
        r = df[[a, b]].corr(method="spearman").iloc[0, 1]
        ax.set_xlabel(a); ax.set_ylabel(b); ax.text(.03, .95, f"Spearman {r:.2f}", transform=ax.transAxes, va="top", fontsize=11); vs.panel_label(ax, l)
    ax = axes[1, 1]
    ax.hist(dev.dropna(), bins=60, color=vs.PALETTE["teal"], edgecolor="white", lw=.4)
    med, sd = thr
    ax.axvline(med + 5 * sd, color=vs.PALETTE["red"], ls="--"); ax.axvline(med - 5 * sd, color=vs.PALETTE["red"], ls="--")
    ax.set_xlabel("ln( total_spend / (n_orders x avg_order_value) )"); ax.set_ylabel("count"); ax.set_yscale("log"); vs.panel_label(ax, "d")
    vs.finalize_figure(fig, FIG, "fig10_key_relationships")


def fig_outliers(F, per_feat):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    cnt = F.sum().sort_values()
    axes[0].barh(cnt.index, cnt.values, color=vs.PALETTE["blue"], edgecolor="white", lw=.6)
    for y, v in enumerate(cnt.values): axes[0].text(v + 1, y, str(int(v)), va="center")
    axes[0].set_xlabel(f"Rows flagged (of {len(F)})"); vs.panel_label(axes[0], "a")
    w = .27; x = np.arange(len(per_feat))
    for k, (col, c, h) in enumerate(zip(["domain", "tukey_1.5", "tukey_3"], vs.SERIES[:3], vs.HATCHES[:3])):
        axes[1].bar(x + (k - 1) * w, per_feat[col], w, label=col, color=c, hatch=h, edgecolor="white", lw=.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(per_feat.index, rotation=40, ha="right"); axes[1].set_ylabel("Cells flagged"); axes[1].legend(); vs.panel_label(axes[1], "b")
    vs.finalize_figure(fig, FIG, "fig11_outlier_detection")


def fig_distance(df, num, rawz, rob, share, hard, d_a):
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    ax = axes[0, 0]; s = (100 * share).sort_values()
    ax.barh(s.index, s.values, color=vs.PALETTE["blue"], edgecolor="white", lw=.6)
    for y, v in enumerate(s.values): ax.text(v + .5, y, f"{v:.2f}%", va="center", fontsize=10)
    ax.tick_params(axis="y", labelsize=11); ax.set_xlabel("Share of total variance, unscaled (%)"); ax.set_xlim(0, max(s.values) * 1.35); vs.panel_label(ax, "a")
    ax = axes[0, 1]
    for name, Z, c, ls in [("z-score on raw", rawz, vs.PALETTE["red"], "--"), ("log1p + median/IQR", rob, vs.PALETTE["blue"], "-")]:
        k = knn_dist(Z); xs = np.sort(k); ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, ls=ls, label=name)
    ax.set_xscale("log"); ax.set_xlabel("distance to 5th nearest neighbour"); ax.set_ylabel("ECDF"); ax.legend(loc="lower right", fontsize=11); vs.panel_label(ax, "b")
    ax = axes[1, 0]
    for Z, ttl, l, c in [(rawz, "z-score on raw", "c", vs.PALETTE["red"]), (rob, "log1p + median/IQR", "d", vs.PALETTE["blue"])]:
        pass
    for ax, Z, ttl, l in [(axes[1, 0], rawz, "z-score on raw", "c"), (axes[1, 1], rob, "log1p + median/IQR", "d")]:
        P, var = pca2(Z)
        ax.scatter(P[~hard.to_numpy(), 0], P[~hard.to_numpy(), 1], s=26, color=vs.PALETTE["blue"], alpha=.7, edgecolor="none", label="not flagged")
        ax.scatter(P[hard.to_numpy(), 0], P[hard.to_numpy(), 1], s=80, marker="^", color=vs.PALETTE["red"], edgecolor="white", lw=.6, label="rule/identity violation")
        ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)"); ax.set_title(ttl, fontsize=12); vs.panel_label(ax, l)
        if l == "c": ax.legend()
    vs.finalize_figure(fig, FIG, "fig12_distance_and_density")


# ---------------------------------------------------------------------------------------------- PHẦN B
def b_compare(df3, num, log_cols, F, hard):
    d0, d1, d2 = (pd.read_csv(DATA / f) for f in ["data_clean_reference.csv", "data_after_noise.csv", "data_after_outliers.csv"])
    truth = pd.read_csv(HID / "ground_truth.csv")
    inj = pd.read_csv(HID / "injection_log.csv")
    stages = dict(zip(STAGES, [d0, d1, d2, df3]))
    log.info("\n" + "=" * 80 + "\nPHẦN B - ĐỐI CHIẾU VỚI DỮ LIỆU NỀN VÀ NHẬT KÝ CHÈN LỖI (đã chốt Phần A trước khi đọc)\n" + "=" * 80)
    stats_rows = []
    for st, d in stages.items():
        for c in num:
            x = d[c].astype(float)
            stats_rows.append(dict(stage=st, column=c, mean=x.mean(), std=x.std(), median=x.median(), iqr=x.quantile(.75) - x.quantile(.25), skew=x.skew(), min=x.min(), max=x.max(), n_missing=int(x.isna().sum())))
    S = pd.DataFrame(stats_rows)
    save_table(S.round(4), "b1_stage_summary_stats", index=False)
    # noise D0->D1
    nz = []
    for c in ["avg_order_value", "recency_days", "avg_session_minutes", "sessions_per_month"]:
        a, b = d0[c].astype(float), d1[c].astype(float)
        delta = b - a
        nz.append(dict(column=c, mean_change_pct=100 * (b.mean() / a.mean() - 1), std_change_pct=100 * (b.std() / a.std() - 1), noise_sd=delta.std(),
                       snr_var_ratio=a.var() / delta.var(), corr_D0_D1=a.corr(b), ks_stat=stats.ks_2samp(a, b).statistic, wasserstein=stats.wasserstein_distance(a, b)))
    NZ = pd.DataFrame(nz).set_index("column")
    show("B1. Tác động của NHIỄU (D0 -> D1)", NZ.round(4)); save_table(NZ.round(5), "b1_noise_effect")
    # outliers D1->D2
    oe = pd.DataFrame({c: dict(mean_ratio=d2[c].mean() / d1[c].mean(), std_ratio=d2[c].std() / d1[c].std(), skew_D1=d1[c].skew(), skew_D2=d2[c].skew(),
                               max_D1=d1[c].max(), max_D2=d2[c].max(), n_cells_changed=int((d1[c].astype(float) != d2[c].astype(float)).sum())) for c in num}).T
    show("B1. Tác động của NGOẠI LAI (D1 -> D2)", oe.round(3)); save_table(oe.round(4), "b1_outlier_effect")
    # missing D2->D3
    ms = []
    for c in ["age", "avg_session_minutes"]:
        obs = ~df3[c].isna()
        ms.append(dict(column=c, mean_full_D2=d2[c].mean(), mean_observed_D3=d3m(df3, c), mean_clean_D0=d0[c].mean(), median_full_D2=d2[c].median(), median_observed_D3=df3[c].median(), median_clean_D0=d0[c].median(),
                       std_full_D2=d2[c].std(), std_observed_D3=df3[c].std()))
    ME = pd.DataFrame(ms).set_index("column")
    show("B1. Tác động của GIÁ TRỊ THIẾU (D2 -> D3) lên thống kê quan sát được", ME.round(3)); save_table(ME.round(4), "b1_missing_effect")
    rows_m = []
    for dv, g in d2.assign(m=df3["avg_session_minutes"].isna()).groupby("device_type"):
        rows_m.append(dict(device=dv, median_all=g["avg_session_minutes"].median(), median_of_missing_cells=g.loc[g.m, "avg_session_minutes"].median(),
                           median_of_observed_cells=g.loc[~g.m, "avg_session_minutes"].median(), n_missing=int(g.m.sum())))
    mm = pd.DataFrame(rows_m).set_index("device")
    show("B1. avg_session_minutes: TRUNG VỊ giá trị thật (D2) của ô bị thiếu so với ô còn lại, theo thiết bị (dùng trung vị vì dòng 1440 phút làm méo trung bình)", mm.round(2))
    # correlation change
    pear = {st: d[num].astype(float).corr() for st, d in stages.items()}
    spear = {st: d[num].astype(float).corr(method="spearman") for st, d in stages.items()}
    dP, dS = pear["D3 +missing"] - pear["D0 clean"], spear["D3 +missing"] - spear["D0 clean"]
    iu = np.triu_indices(len(num), 1)
    log.info("\nB1. Thay đổi tương quan D3 so với D0: Pearson max|Δ|=%.3f, mean|Δ|=%.3f | Spearman max|Δ|=%.3f, mean|Δ|=%.3f",
             np.abs(dP.to_numpy()[iu]).max(), np.abs(dP.to_numpy()[iu]).mean(), np.abs(dS.to_numpy()[iu]).max(), np.abs(dS.to_numpy()[iu]).mean())
    big = pd.DataFrame([dict(a=num[i], b=num[j], pearson_D0=pear["D0 clean"].iloc[i, j], pearson_D3=pear["D3 +missing"].iloc[i, j], spearman_D0=spear["D0 clean"].iloc[i, j], spearman_D3=spear["D3 +missing"].iloc[i, j]) for i, j in zip(*iu)])
    big["abs_dPearson"] = (big.pearson_D3 - big.pearson_D0).abs()
    show("B1. 8 cặp có Pearson thay đổi nhiều nhất (D0 -> D3)", big.sort_values("abs_dPearson", ascending=False).head(8).round(3))
    # distances / structure across stages
    lens = {st: {k: make_lens(d, num, k, log_cols) for k in ("raw_z", "robust")} for st, d in stages.items()}
    seg = truth["true_segment"].to_numpy()
    res = []
    for st in STAGES:
        for k in ("raw_z", "robust"):
            X0, Xk = lens[STAGES[0]][k], lens[st][k]
            n0, nk = knn_sets(X0), knn_sets(Xk)
            res.append(dict(stage=st, lens=k, neighbour_preserved=float(np.mean([len(a & b) / 5 for a, b in zip(n0, nk)])), dist_spearman=float(stats.spearmanr(pdist(X0), pdist(Xk))[0]),
                            hopkins=hopkins(Xk, SEED)[0], silhouette_true_labels=float(silhouette_score(Xk, seg))))
    R = pd.DataFrame(res)
    show("B2. Cấu trúc khoảng cách/mật độ qua từng giai đoạn (so với D0; silhouette dùng nhãn ẩn chỉ để chẩn đoán)", R.round(3)); save_table(R.round(4), "b2_structure_by_stage", index=False)
    # audit of blind detections
    inj_ids = inj.loc[inj.stage == "outlier"].drop_duplicates("customer_id").set_index("customer_id")["outlier_class"]
    ids = df3["customer_id"]
    truth_set = set(inj_ids.index)
    au = []
    for m in F.columns:
        fl = set(ids[F[m]])
        tp = len(fl & truth_set)
        au.append(dict(method=m, flagged=len(fl), true_injected_caught=tp, precision=tp / len(fl) if fl else np.nan, recall=tp / len(truth_set),
                       recall_error=len(fl & set(inj_ids[inj_ids == "error"].index)) / (inj_ids == "error").sum(), recall_genuine=len(fl & set(inj_ids[inj_ids == "genuine"].index)) / (inj_ids == "genuine").sum()))
    flh = set(ids[hard]); au.append(dict(method="hard_errors(rules|identity)", flagged=len(flh), true_injected_caught=len(flh & truth_set), precision=len(flh & truth_set) / len(flh), recall=len(flh & truth_set) / len(truth_set),
                                         recall_error=len(flh & set(inj_ids[inj_ids == "error"].index)) / (inj_ids == "error").sum(), recall_genuine=len(flh & set(inj_ids[inj_ids == "genuine"].index)) / (inj_ids == "genuine").sum()))
    A = pd.DataFrame(au).set_index("method")
    show("B3. Kiểm chứng phát hiện mù: so với 15 khách thật sự bị chèn ngoại lai (8 hợp lệ + 7 lỗi)", A.round(3)); save_table(A.round(4), "b3_outlier_detection_audit")
    log.info("Quan sát độ lệch đẳng thức: sd bền quan sát = %.3f; sd kỳ vọng từ nhiễu nhân avg_order_value 0.12 và biến thiên total 0.10 = %.3f", KEY["identity_sd_robust"], float(np.hypot(.12, .10)))
    return dict(stages=stages, NZ=NZ, oe=oe, pear=pear, spear=spear, dP=dP, dS=dS, R=R, truth=truth, lens=lens, A=A, S=S)


def d3m(df, c):
    return float(df[c].mean())


def fig_compare(B, num):
    NZ, stages = B["NZ"], B["stages"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.4))
    for ax, c, l in zip(axes, NZ.index, "abcd"):
        for st in STAGES[:2]:
            x = np.sort(stages[st][c].astype(float)); ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=vs.STAGE_COLORS[st], ls=vs.STAGE_LINESTYLES[st], label=st)
        if c in ("avg_order_value", "avg_session_minutes", "sessions_per_month"): ax.set_xscale("log")
        ax.set_title(f"{c}\nKS={NZ.loc[c, 'ks_stat']:.2f}, SNR={NZ.loc[c, 'snr_var_ratio']:.0f}", fontsize=11); ax.set_ylabel("ECDF" if l == "a" else ""); vs.panel_label(ax, l)
    axes[0].legend(loc="lower right")
    vs.finalize_figure(fig, FIG, "fig13_noise_effect_ecdf")

    oe = B["oe"]
    fig, axes = plt.subplots(1, 2, figsize=(19, 5.8))
    for ax, col, ttl, l in [(axes[0], "std_ratio", "std(D2) / std(D1)", "a"), (axes[1], "max_D2", "max in D2 vs D1", "b")]:
        x = np.arange(len(num))
        if col == "std_ratio":
            ax.bar(x, oe[col].astype(float), color=vs.PALETTE["violet"], edgecolor="white", lw=.6, hatch="..")
            ax.axhline(1, color=vs.NEUTRAL["ink"], lw=1.2); ax.set_yscale("log")
        else:
            ax.bar(x - .2, oe["max_D1"].astype(float), .4, color=vs.PALETTE["red"], edgecolor="white", lw=.6, label="D1 +noise", hatch="//")
            ax.bar(x + .2, oe["max_D2"].astype(float), .4, color=vs.PALETTE["teal"], edgecolor="white", lw=.6, label="D2 +outliers", hatch=".."); ax.set_yscale("log"); ax.legend()
        ax.set_xticks(x); ax.set_xticklabels(num, rotation=40, ha="right"); ax.set_ylabel(ttl); vs.panel_label(ax, l)
    vs.finalize_figure(fig, FIG, "fig14_outlier_effect")

    fig, axes = plt.subplots(1, 3, figsize=(23, 9), gridspec_kw={"width_ratios": [1, 1, 0.04]})
    lim = float(np.abs(B["dP"].to_numpy()).max())
    for ax, M, t, l in [(axes[0], B["dP"], "Pearson: D3 minus D0", "a"), (axes[1], B["dS"], "Spearman: D3 minus D0", "b")]:
        im = ax.imshow(M.to_numpy(), cmap=vs.CMAP_DIV, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=45, ha="right"); ax.set_yticks(range(len(num))); ax.set_yticklabels(num)
        for i in range(len(num)):
            for j in range(len(num)): ax.text(j, i, f"{M.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(M.iloc[i, j]) > .6 * lim else vs.NEUTRAL["ink"])
        ax.set_title(t); vs.panel_label(ax, l)
        for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im, cax=axes[2], label="change in correlation")
    vs.finalize_figure(fig, FIG, "fig15_correlation_change", pad=1.5)

    R = B["R"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    lens_style = {"raw_z": ("z-score on raw", "//"), "robust": ("log1p + median/IQR", "")}
    for ax, metric, ttl, l in [(axes[0, 0], "neighbour_preserved", "5-NN neighbours preserved vs D0", "a"), (axes[0, 1], "dist_spearman", "Spearman of pairwise distances vs D0", "b")]:
        x = np.arange(1, 4)
        for k, (lbl, h) in zip(["raw_z", "robust"], lens_style.values()):
            sub = R[(R.lens == k) & (R.stage != STAGES[0])]
            ax.bar(x + (-.2 if k == "raw_z" else .2), sub[metric], .38, color=[vs.STAGE_COLORS[s] for s in sub.stage], hatch=h, edgecolor="white", lw=.6)
            for xi, v in zip(x + (-.2 if k == "raw_z" else .2), sub[metric]): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(STAGES[1:]); ax.set_ylim(0, 1.12); ax.set_ylabel(ttl); vs.panel_label(ax, l)
    from matplotlib.patches import Patch
    axes[0, 0].legend(handles=[Patch(facecolor=vs.NEUTRAL["light"], hatch=h, edgecolor=vs.NEUTRAL["ink"], label=lbl) for lbl, h in lens_style.values()], loc="upper right")
    for ax, metric, ttl, l in [(axes[1, 0], "hopkins", "Hopkins statistic", "c"), (axes[1, 1], "silhouette_true_labels", "Silhouette of hidden segments (diagnostic)", "d")]:
        for k, (lbl, _) in lens_style.items():
            sub = R[R.lens == k]
            ax.plot(range(4), sub[metric], marker="o" if k == "raw_z" else "s", ls="--" if k == "raw_z" else "-", color=vs.PALETTE["red"] if k == "raw_z" else vs.PALETTE["blue"], label=lbl)
        ax.set_xticks(range(4)); ax.set_xticklabels(STAGES, rotation=15); ax.set_ylabel(ttl); vs.panel_label(ax, l)
    axes[1, 0].legend()
    vs.finalize_figure(fig, FIG, "fig16_structure_across_stages")

    truth, lens = B["truth"], B["lens"]
    seg = truth["true_segment"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.6))
    P_ref = pca2(lens[STAGES[0]]["robust"])[0]
    for ax, st, l in [(axes[0], STAGES[0], "a"), (axes[1], STAGES[3], "b")]:
        P, var = pca2(lens[st]["robust"])
        for j in range(2):  # dấu vector riêng là tùy ý: căn theo D0 để so sánh được
            if np.corrcoef(P_ref[:, j], P[:, j])[0, 1] < 0:
                P[:, j] *= -1
        for s, c, m in zip(SEGMENTS, vs.SERIES, ["o", "s", "^", "D"]):
            ax.scatter(P[seg == s, 0], P[seg == s, 1], s=34, color=c, marker=m, alpha=.8, edgecolor="none", label=s)
        ax.set_title(f"{st} (log1p + median/IQR lens)", fontsize=12); ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)"); vs.panel_label(ax, l)
    axes[0].legend()
    vs.finalize_figure(fig, FIG, "fig17_pca_hidden_segments")


# ---------------------------------------------------------------------------------------------- main
def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOGD / "02_data_understanding.log", mode="w", encoding="utf-8")])
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    vs.apply_publication_style()
    log.info("=" * 80 + "\nPHẦN A - PHÂN TÍCH MÙ: chỉ đọc data_raw_corrupted.csv và data_dictionary.csv\n" + "=" * 80)
    df = pd.read_csv(DATA / "data_raw_corrupted.csv")
    dd = pd.read_csv(DATA / "data_dictionary.csv")
    num = dd.loc[dd["dtype"].isin(["int", "float"]), "attribute"].tolist()
    a1_overview(df, dd, num)
    desc, log_cols = a2_descriptives(df, dd, num)
    V = a3_domain(df, num)
    M, miss_cols, dev = a4_missing(df, num)
    sp, pe, cv, eps, vif = a5_relations(df, num)
    F, hard, ident, thr, per_feat = a6_outliers(df, num, log_cols, V)
    share, rawz, rob, d_a = a7_distance(df, num, log_cols, hard)
    fig_missing(df, M, miss_cols, dev); fig_hist_box(df, num, log_cols); fig_categorical(df, num); fig_correlation(sp, pe, cv, eps)
    fig_scatter(df, ident, thr); fig_outliers(F, per_feat); fig_distance(df, num, rawz, rob, share, hard, d_a)
    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(KEY, open(TAB / "b_key_numbers_partA.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    B = b_compare(df, num, log_cols, F, hard)
    fig_compare(B, num)
    json.dump(KEY, open(TAB / "b_key_numbers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log.info("\nĐã ghi %d hình vào %s", len(list(FIG.glob('*.png'))), FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
