#!/usr/bin/env python3
"""Câu d - Data Preparation: mã hóa, chuẩn hóa, chọn thuộc tính, (tùy chọn) giảm chiều cho phân cụm dựa trên khoảng cách.

Đầu vào: data/02_preprocessed (đã đóng băng; script kiểm tra mã băm) + data dictionary. Không đọc D0-D2 hay thư mục dữ liệu ẩn.
Ma trận chính (X_main): log1p cho cột lệch phải -> StandardScaler; thứ tự (hạng) -> mã 0/1/2 -> z-score; nhị phân -> z-score;
biến định danh -> one-hot KHÔNG chuẩn hóa nhưng nhân w = 1/sqrt(1 - sum p^2) để E[d^2] của mỗi biến bằng một cột số chuẩn hóa (bằng 2);
mức 'unknown' của preferred_category -> hàng toàn số 0 (cách đều mọi nhãn, không tạo trục riêng); bỏ total_spend_12m (dư thừa); không PCA.
Các phiên bản thay thế lưu riêng: full_features, pca90, cat_unknown_col, cat_knn7, cat_mode, no_category, Gower.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_style as vs  # noqa: E402
from dm_utils import select_log_cols  # noqa: E402
from prep_utils import scale_by  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, PRE = ROOT / "data" / "01_generated", ROOT / "data" / "02_preprocessed"
OUT = ROOT / "data" / "03_prepared"
FIG, TAB, LOGD = ROOT / "figures" / "04_preparation", ROOT / "reports" / "tables", ROOT / "logs"
SEED = 42
BEFORE, AFTER = vs.PALETTE["violet"], vs.PALETTE["teal"]
TIERS = ["bronze", "silver", "gold"]
log = logging.getLogger("d")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def show(title, obj):
    log.info("\n--- %s ---\n%s", title, obj.to_string() if hasattr(obj, "to_string") else obj)


def save_table(df, name, index=False):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=index)


def write_csv(df, path, **kw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.chmod(path, 0o644)
    df.to_csv(path, **kw)


class Checker:
    def __init__(self):
        self.rows = []

    def __call__(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        log.info("[%s] %s %s", "PASS" if ok else "FAIL", name, detail)

    @property
    def all_ok(self):
        return all(ok for _, ok, _ in self.rows)


def verify_input_freeze():
    man = json.loads((PRE / "FROZEN_SHA256.json").read_text(encoding="utf-8"))
    bad = [rel for rel, h in man["frozen_files"].items() if sha256(ROOT / rel) != h]
    if bad:
        raise SystemExit(f"TỪ CHỐI: dữ liệu câu c không khớp mã băm đã đóng băng: {bad}")
    return man


def w_nominal(s):
    """w = 1/sqrt(1 - sum p^2): E[d^2] giữa hai khách ngẫu nhiên của biến nominal one-hot x w bằng đúng 2 (như một cột z-score)."""
    p = s.value_counts(normalize=True).to_numpy()
    return float(1 / np.sqrt(1 - (p ** 2).sum()))


def make_pipeline(num_log, num_lin, w_dev, w_cat, cat_levels, use_cat=True, bin_col="app_installed"):
    t = [("num_log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("z", StandardScaler())]), num_log),
         ("num_lin", StandardScaler(), num_lin),
         ("tier", Pipeline([("ord", OrdinalEncoder(categories=[TIERS])), ("z", StandardScaler())]), ["membership_tier"]),
         ("app", StandardScaler(), [bin_col]),
         ("device", Pipeline([("ohe", OneHotEncoder(categories=[["desktop", "mobile", "tablet"]], handle_unknown="ignore", sparse_output=False)),
                              ("w", FunctionTransformer(scale_by, kw_args={"w": w_dev}, feature_names_out="one-to-one"))]), ["device_type"])]
    if use_cat:
        t.append(("category", Pipeline([("ohe", OneHotEncoder(categories=[cat_levels], handle_unknown="ignore", sparse_output=False)),
                                        ("w", FunctionTransformer(scale_by, kw_args={"w": w_cat}, feature_names_out="one-to-one"))]), ["preferred_category"]))
    return ColumnTransformer(t, remainder="drop", verbose_feature_names_out=True).set_output(transform="pandas")


def build(proc, num_log, num_lin, cat_col="preferred_category", cat_levels=None, use_cat=True, unknown_as_level=False):
    d = proc.copy()
    levels = cat_levels or sorted(l for l in d[cat_col].unique() if l != "unknown")
    if unknown_as_level:
        levels = sorted(d[cat_col].unique())
        w_cat = w_nominal(d[cat_col])
    else:
        w_cat = w_nominal(d.loc[d[cat_col] != "unknown", cat_col])
    d["preferred_category"] = d[cat_col]
    pipe = make_pipeline(num_log, num_lin, w_nominal(d["device_type"]), w_cat, levels, use_cat)
    X = pipe.fit_transform(d)
    return pipe, X, dict(w_device=w_nominal(d["device_type"]), w_category=w_cat, cat_levels=levels)


def gower(proc, num_log, num_lin):
    n = len(proc)
    D, W = np.zeros((n, n)), np.zeros((n, n))
    cols = {c: np.log1p(proc[c].to_numpy(float)) for c in num_log}
    cols.update({c: proc[c].to_numpy(float) for c in num_lin})
    cols["membership_tier"] = proc["membership_tier"].map({t: i for i, t in enumerate(TIERS)}).to_numpy(float)
    for x in cols.values():
        r = x.max() - x.min()
        D += np.abs(x[:, None] - x[None, :]) / r
        W += 1
    for c in ("app_installed", "device_type"):
        x = proc[c].to_numpy()
        D += (x[:, None] != x[None, :]); W += 1
    x = proc["preferred_category"].to_numpy()
    ok = (x != "unknown")
    m = ok[:, None] & ok[None, :]
    D += (x[:, None] != x[None, :]) * m; W += m  # Gower bỏ qua biến khi một bên không có thông tin
    return D / W


def vif_table(df, num, log_cols):
    X = df[num].astype(float).copy()
    for c in log_cols:
        X[c] = np.log1p(X[c])
    X["tier"] = df["membership_tier"].map({t: i for i, t in enumerate(TIERS)})
    X["app"] = df["app_installed"].astype(float)
    out = {}
    for c in X.columns:
        y = X[c].to_numpy(); A = np.c_[np.ones(len(X)), X.drop(columns=c).to_numpy()]
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        r2 = 1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = (r2, 1 / (1 - r2))
    return pd.DataFrame(out, index=["R2", "VIF"]).T


def influence(Z):
    d = Z - np.median(Z, 0)
    ss = (d ** 2).sum(1)
    v = (d ** 2).sum(0)
    return dict(top5_rows_share_of_SS=np.sort(ss)[::-1][:5].sum() / ss.sum(), top1_feature_share=v.max() / v.sum(), max_abs_value=np.abs(Z).max(),
                cells_beyond_4sd=int((np.abs((Z - np.median(Z, 0)) / np.where(Z.std(0) > 0, Z.std(0), 1)) > 4).sum()))


def nn_sets(Z, k=5):
    return [set(r) for r in cKDTree(Z).query(Z, k=k + 1)[1][:, 1:]]


def nn_agree(Za, Zb):
    return float(np.mean([len(a & b) / 5 for a, b in zip(nn_sets(Za), nn_sets(Zb))]))


def mean_sq_pair(M):
    M = np.asarray(M, float)
    return float(2 * M.var(axis=0).sum())  # E[d^2] giữa hai điểm độc lập = 2 x tổng phương sai


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOGD / "04_prepare.log", mode="w", encoding="utf-8")])
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    vs.apply_publication_style()
    man = verify_input_freeze()
    ck = Checker()
    dd = pd.read_csv(DATA / "data_dictionary.csv")
    num = dd.loc[dd["dtype"].isin(["int", "float"]), "attribute"].tolist()
    proc = pd.read_csv(PRE / "data_after_preprocessing.csv")
    variants = pd.read_csv(PRE / "variants" / "preferred_category_variants.csv")
    log.info("=" * 80 + "\nCÂU D - CHUẨN BỊ DỮ LIỆU. Đầu vào đã đóng băng (câu c), mã băm khớp. Shape: %s\n" + "=" * 80, proc.shape)
    log_all, _ = select_log_cols(proc, dd, num)
    DROP = ["total_spend_12m"]
    num_log = [c for c in log_all if c not in DROP]
    num_lin = [c for c in num if c not in log_all]
    log.info("Cột log1p: %s | cột tuyến tính: %s | bỏ vì dư thừa: %s", num_log, num_lin, DROP)

    # ---------------- bằng chứng chọn phương án (chỉ từ dữ liệu đã xử lý)
    vif_all, vif_drop = vif_table(proc, num, log_all), vif_table(proc.drop(columns=DROP), [c for c in num if c not in DROP], num_log)
    V = vif_all.join(vif_drop, lsuffix="_with_total", rsuffix="_without_total")
    show("1. Dư thừa trên thang log: R2/VIF khi có và không có total_spend_12m", V.round(3))
    save_table(V.round(4), "d_vif_with_without_total", index=True)
    X10 = proc[num].astype(float)
    XL10 = X10.copy()
    for c in log_all:
        XL10[c] = np.log1p(XL10[c])
    sc_rows = {}
    for lab, M in (("raw", X10), ("log1p", XL10)):
        for nm, sc in (("StandardScaler", StandardScaler()), ("MinMaxScaler", MinMaxScaler()), ("RobustScaler", RobustScaler())):
            sc_rows[f"{lab} + {nm}"] = influence(sc.fit_transform(M))
    SC = pd.DataFrame(sc_rows).T
    show("2. So sánh scaler (10 cột số): mức chi phối của vài dòng/cột", SC.round(3))
    save_table(SC.round(4), "d_scaler_comparison", index=True)

    # ---------------- ma trận chính và các phiên bản
    pipe, X, info = build(proc, num_log, num_lin)
    X.columns = [c.split("__", 1)[1] if "__" in c else c for c in X.columns]
    log.info("w device = %.4f, w category = %.4f (tính trên %d nhãn đã biết)", info["w_device"], info["w_category"], int((proc["preferred_category"] != "unknown").sum()))
    ids = proc["customer_id"]
    variants_X, extra = {}, {}
    _, Xf, inf_f = build(proc, log_all, num_lin); Xf.columns = [c.split("__", 1)[1] for c in Xf.columns]; variants_X["full_features"] = Xf
    _, Xu, _ = build(proc, num_log, num_lin, unknown_as_level=True); Xu.columns = [c.split("__", 1)[1] for c in Xu.columns]; variants_X["cat_unknown_col"] = Xu
    for nm, col in (("cat_knn7", "preferred_category_knn7"), ("cat_mode", "preferred_category_mode")):
        d2 = proc.copy(); d2["preferred_category"] = variants[col].to_numpy()
        _, Xv, _ = build(d2, num_log, num_lin, cat_levels=sorted(variants[col].unique())); Xv.columns = [c.split("__", 1)[1] for c in Xv.columns]; variants_X[nm] = Xv
    _, Xn, _ = build(proc, num_log, num_lin, use_cat=False); Xn.columns = [c.split("__", 1)[1] for c in Xn.columns]; variants_X["no_category"] = Xn
    pca = PCA(n_components=0.90, svd_solver="full", random_state=0).fit(X.to_numpy())
    Xp = pd.DataFrame(pca.transform(X.to_numpy()), columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
    variants_X["pca90"] = Xp
    G = gower(proc, num_log, num_lin)

    # ---------------- thống kê trước/sau
    rows = []
    for c in num_log + num_lin + DROP:
        b = proc[c].astype(float)
        a = (np.log1p(b) if c in log_all else b)
        z = (a - a.mean()) / a.std(ddof=0)
        rows.append(dict(feature=c, dropped=c in DROP, transform="log1p + z-score" if c in log_all else "z-score", before_mean=b.mean(), before_std=b.std(), before_min=b.min(), before_max=b.max(),
                         before_skew=b.skew(), after_mean=z.mean(), after_std=z.std(ddof=0), after_min=z.min(), after_max=z.max(), after_skew=z.skew()))
    SB = pd.DataFrame(rows)
    show("3. Thang đo và hình dạng phân phối trước/sau (cột số)", SB.round(3))
    save_table(SB.round(4), "d_scale_before_after")
    # khối one-hot
    dev, cat_noun = pd.get_dummies(proc["device_type"]).astype(float), pd.get_dummies(proc["preferred_category"]).astype(float)
    catk = cat_noun.drop(columns="unknown")
    ZN = StandardScaler().fit_transform(np.c_[np.log1p(proc[num_log]), proc[num_lin]])
    tz = X["tier__membership_tier"] if "tier__membership_tier" in X.columns else X.filter(like="membership_tier").iloc[:, 0]
    tier_app = X.filter(regex="membership_tier|app_installed")
    def share(blocks):
        t = sum(np.var(np.asarray(v, float), axis=0).sum() for v in blocks.values())
        return {k: float(np.var(np.asarray(v, float), axis=0).sum() / t) for k, v in blocks.items()}
    wd, wc = info["w_device"], info["w_category"]
    BS = pd.DataFrame({
        "A. z-score ca dummy": share({"numeric": ZN, "tier+app": tier_app, "device": StandardScaler().fit_transform(dev), "category": StandardScaler().fit_transform(cat_noun)}),
        "B. dummy 0/1 khong scale": share({"numeric": ZN, "tier+app": tier_app, "device": dev.to_numpy(), "category": catk.to_numpy()}),
        "C. dummy x w (chinh)": share({"numeric": ZN, "tier+app": tier_app, "device": dev.to_numpy() * wd, "category": catk.to_numpy() * wc})})
    show("4. Tỉ trọng phương sai theo khối trong ma trận thiết kế (mã hóa one-hot)", BS.round(3))
    save_table(BS.round(4), "d_block_variance_shares", index=True)
    per_var = {"device_type (3 cot)": mean_sq_pair(X.filter(like="device_type")), "preferred_category (5 cot)": mean_sq_pair(X.filter(like="preferred_category")),
               "membership_tier (1 cot)": mean_sq_pair(X.filter(like="membership_tier")), "app_installed (1 cot)": mean_sq_pair(X.filter(like="app_installed")),
               "mot cot so (trung binh)": float(np.mean([mean_sq_pair(X[[c]]) for c in num_log + num_lin]))}
    show("4b. E[d^2] giữa hai khách ngẫu nhiên theo từng biến (mục tiêu = 2 cho mỗi biến)", pd.Series(per_var, name="E_d2").round(3))
    # tỉ trọng khoảng cách theo thuộc tính gốc: trước (số thô) / sau (ma trận chính)
    before = (X10.var(ddof=0) / X10.var(ddof=0).sum()).rename("before_share")
    orig = {c: [c] for c in num_log + num_lin}
    after_var = {c: mean_sq_pair(X[orig[c]]) for c in orig}
    after_var.update({"membership_tier": per_var["membership_tier (1 cot)"], "app_installed": per_var["app_installed (1 cot)"], "device_type": per_var["device_type (3 cot)"],
                      "preferred_category": per_var["preferred_category (5 cot)"]})
    after = pd.Series(after_var); after = (after / after.sum()).rename("after_share")
    DS = pd.concat([before, after], axis=1).fillna(0)
    show("5. Tỉ trọng bình phương khoảng cách theo thuộc tính gốc, trước (số thô, không scale) và sau chuẩn bị", DS.round(4))
    save_table(DS.round(5), "d_distance_share_by_feature", index=True)

    # hình học các phiên bản so với ma trận chính
    Xm = X.to_numpy(); dm = pdist(Xm)
    geo = []
    for nm, Xv in variants_X.items():
        geo.append(dict(variant=nm, n_columns=Xv.shape[1], nn5_agree_with_main=nn_agree(Xm, Xv.to_numpy()), dist_spearman_with_main=float(stats.spearmanr(dm, pdist(Xv.to_numpy()))[0])))
    from scipy.spatial.distance import squareform
    iu = np.triu_indices(len(G), 1)
    gk = [set(np.argsort(G[i] + np.eye(len(G))[i] * 9)[:5]) for i in range(len(G))]
    geo.append(dict(variant="gower_distance", n_columns=np.nan, nn5_agree_with_main=float(np.mean([len(a & b) / 5 for a, b in zip(nn_sets(Xm), gk)])), dist_spearman_with_main=float(stats.spearmanr(dm, G[iu])[0])))
    GEO = pd.DataFrame(geo)
    show("6. Hình học của các phiên bản so với ma trận chính", GEO.round(4))
    save_table(GEO.round(5), "d_variants_geometry")
    cum = np.cumsum(PCA().fit(Xm).explained_variance_ratio_)
    pc_tab = pd.DataFrame({"k": range(1, len(cum) + 1), "cum_variance": cum})
    pc_tab["nn5_agree"] = [nn_agree(Xm, PCA(k).fit_transform(Xm)) for k in pc_tab["k"]]
    show("6b. PCA: phương sai tích lũy và độ giữ lân cận 5-NN", pc_tab.round(3))
    save_table(pc_tab.round(4), "d_pca_curve")
    log.info("PCA 90%% phương sai: %d PC; 5-NN giữ lại %.3f", pca.n_components_, float(pc_tab.loc[pca.n_components_ - 1, "nn5_agree"]))

    # ---------------- kiểm tra nghiệm thu
    feat_in = [c for c in proc.columns if c != "customer_id"]
    wd, wc = info["w_device"], info["w_category"]
    meta = []
    for col in X.columns:
        base = col
        kind = "one-hot x w" if ("device_type" in col or "preferred_category" in col) else ("log1p + z-score" if any(col == c for c in num_log) else "z-score" if True else "")
        src = col if col in feat_in else next(c for c in feat_in if col.startswith(c + "_"))
        mean = std = np.nan
        if src in num_log: s = np.log1p(proc[src]); mean, std = s.mean(), s.std(ddof=0)
        elif src in num_lin or src == "app_installed": mean, std = proc[src].mean(), proc[src].std(ddof=0)
        elif src == "membership_tier": s = proc[src].map({t: i for i, t in enumerate(TIERS)}); mean, std = s.mean(), s.std(ddof=0)
        meta.append(dict(column=col, source_feature=src, transform=kind, mean_used=mean, std_used=std, weight_w=(wd if src == "device_type" else wc if src == "preferred_category" else np.nan),
                         block="numeric" if src in num_log + num_lin else "ordinal" if src == "membership_tier" else "binary" if src == "app_installed" else "nominal"))
    MD = pd.DataFrame(meta)
    log.info("\n" + "=" * 80 + "\nNGHIỆM THU\n" + "=" * 80)
    feat_in = [c for c in proc.columns if c != "customer_id"]
    ck("Ma trận chính 260 hàng, không NaN/inf, không có customer_id hay cờ/metadata", X.shape[0] == 260 and np.isfinite(Xm).all() and not any("customer_id" in c for c in X.columns) and
       not (set(X.columns) & {"n_cells_invalid_repaired", "extreme_tukey3_valid", "lof20_suspect"}), f"shape {X.shape}")
    ck("Số đặc trưng: 14 gốc -> 13 sau chọn lọc (bỏ total_spend) -> 19 cột sau mã hóa", len(feat_in) == 14 and X.shape[1] == 19, f"{len(feat_in)} -> 13 -> {X.shape[1]}")
    zc = [c for c in X.columns if c in num_log + num_lin + ["membership_tier", "app_installed"]]
    ck("Cột z-score có trung bình 0 và độ lệch chuẩn 1", np.allclose(X[zc].mean(), 0, atol=1e-9) and np.allclose(X[zc].std(ddof=0), 1, atol=1e-9), f"{len(zc)} cột")
    ck("E[d^2] của mỗi biến (số, hạng, nhị phân, device) bằng 2; category gần 2 (có 13 'unknown' toàn số 0)",
       abs(per_var["device_type (3 cot)"] - 2) < 1e-6 and abs(per_var["mot cot so (trung binh)"] - 2) < 1e-6 and 1.7 < per_var["preferred_category (5 cot)"] < 2.1,
       f"device {per_var['device_type (3 cot)']:.3f}, category {per_var['preferred_category (5 cot)']:.3f}")
    unk = (proc["preferred_category"] == "unknown").to_numpy()
    ck("13 dòng 'unknown' có khối category toàn số 0; các dòng khác có đúng một cột = w", int(unk.sum()) == 13 and np.allclose(X.filter(like="preferred_category").to_numpy()[unk], 0) and
       np.allclose(np.sort(X.filter(like="preferred_category").to_numpy()[~unk], axis=1)[:, -1], wc))
    man_X = np.c_[np.log1p(proc[num_log]).sub(np.log1p(proc[num_log]).mean()).div(np.log1p(proc[num_log]).std(ddof=0)), proc[num_lin].sub(proc[num_lin].mean()).div(proc[num_lin].std(ddof=0))]
    ck("Tái dựng tay các cột số khớp pipeline", np.allclose(man_X, X[num_log + num_lin].to_numpy()), "")
    pipe2, X2, _ = build(proc, num_log, num_lin)
    ck("Chạy lại cho kết quả giống hệt", np.allclose(X2.to_numpy(), Xm) and list(X2.columns) == [c for c in pipe2.get_feature_names_out()])
    ck("Biến thể không giảm chiều tồn tại song song với phiên bản giảm chiều", X.shape[1] == 19 and variants_X["pca90"].shape[1] == pca.n_components_ < 19 and variants_X["full_features"].shape[1] == 20,
       f"main 19, full_features 20, pca90 {pca.n_components_}")
    non_cat = DS.drop(index=["preferred_category", "total_spend_12m"])["after_share"]
    ck("Mỗi biến số/hạng/nhị phân/device đóng góp cùng một tỉ trọng khoảng cách (chênh < 1e-6); category gần bằng", np.ptp(non_cat.to_numpy()) < 1e-6 and abs(DS.at["preferred_category", "after_share"] / non_cat.iloc[0] - 1) < 0.1,
       f"{non_cat.iloc[0]:.4f} mỗi biến; category {DS.at['preferred_category', 'after_share']:.4f}")
    ck("Siêu dữ liệu: mỗi cột của X_main ánh xạ đúng một thuộc tính gốc, đủ 13 thuộc tính", MD["source_feature"].nunique() == 13 and set(MD["source_feature"]) <= set(feat_in) and len(MD) == 19, "")
    ck("Ma trận Gower đối xứng, đường chéo 0, giá trị trong [0, 1]", np.allclose(G, G.T) and np.allclose(np.diag(G), 0) and G.min() >= 0 and G.max() <= 1, f"max {G.max():.3f}")

    # ---------------- ghi file
    pdir = OUT / "pipeline"
    write_csv(X, OUT / "X_main.csv", index=False)
    write_csv(ids.to_frame(), OUT / "customer_ids.csv", index=False)
    for nm, Xv in variants_X.items():
        write_csv(Xv, OUT / "variants" / f"X_{nm}.csv", index=False)
    np.save(OUT / "variants" / "gower_distance.npy", G)
    pdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, pdir / "prep_pipeline.joblib"); joblib.dump(pca, pdir / "pca90.joblib")
    show("7. Siêu dữ liệu từng cột của ma trận chính", MD.round(4))
    write_csv(MD.round(6), pdir / "feature_metadata.csv", index=False)
    cfg = dict(log1p_columns=num_log, linear_columns=num_lin, dropped_for_redundancy=DROP, ordinal_order=TIERS, w_device=wd, w_category=wc, category_levels=info["cat_levels"],
               unknown_encoding="all-zero category block", scaler="StandardScaler (sau log1p cho cột lệch)", pca90_components=int(pca.n_components_), seed=SEED,
               input_frozen_sha256=man["frozen_files"], versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
               load_note="joblib.load cần src/ trong sys.path để import prep_utils.scale_by")
    (pdir / "prep_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    code = f"import sys; sys.path.insert(0, r'{ROOT / 'src'}'); import joblib, pandas as pd, numpy as np; p = joblib.load(r'{pdir / 'prep_pipeline.joblib'}'); " \
           f"d = pd.read_csv(r'{PRE / 'data_after_preprocessing.csv'}'); X = p.transform(d); print(float(np.abs(X.to_numpy() - pd.read_csv(r'{OUT / 'X_main.csv'}').to_numpy()).max()))"
    diff = float(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip() or "nan")
    ck("Pipeline lưu bằng joblib nạp lại trong tiến trình mới và tái tạo X_main.csv", diff < 1e-9, f"sai khác lớn nhất {diff:.2e}")

    # ---------------- hình
    fig, axes = plt.subplots(4, 5, figsize=(23, 14))
    for k, c in enumerate(num):
        b = proc[c].astype(float); a = (np.log1p(b) if c in log_all else b); z = (a - a.mean()) / a.std(ddof=0)
        r, cc = divmod(k, 5)
        axes[r, cc].hist(b, bins=30, color=BEFORE, edgecolor="white", lw=.4); axes[r, cc].set_title(f"{c}{' (dropped)' if c in DROP else ''}\nBefore: skew {b.skew():.2f}", fontsize=10)
        axes[r + 2, cc].hist(z, bins=30, color=AFTER, edgecolor="white", lw=.4); axes[r + 2, cc].set_title(f"After ({'log1p + ' if c in log_all else ''}z): skew {z.skew():.2f}", fontsize=10)
    vs.finalize_figure(fig, FIG, "figD1_distributions_before_after")
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8))
    for ax, col, ttl, l in ((axes[0], "top5_rows_share_of_SS", "share of total sum of squares held by the top-5 rows", "a"), (axes[1], "top1_feature_share", "share of variance held by the top feature", "b"),
                            (axes[2], "max_abs_value", "max |value| after scaling", "c")):
        v = SC[col]; ax.bar(range(len(v)), v.values, color=[AFTER if i == "log1p + StandardScaler" else vs.NEUTRAL["light"] for i in v.index], edgecolor=vs.NEUTRAL["ink"], lw=.8)
        for b, h in zip(ax.patches, ["", "//", "..", "", "//", ".."]):
            b.set_hatch(h)
        ax.set_xticks(range(len(v))); ax.set_xticklabels(v.index, rotation=40, ha="right"); ax.set_ylabel(ttl); vs.panel_label(ax, l)
        if col == "max_abs_value": ax.set_yscale("log")
    vs.finalize_figure(fig, FIG, "figD2_scaler_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    ax = axes[0]; x = np.arange(len(BS)); w = .26
    for k, (cname, h) in enumerate(zip(BS.columns, ["//", "..", ""])):
        ax.bar(x + (k - 1) * w, BS[cname], w, color=AFTER if k == 2 else vs.NEUTRAL["light"], hatch=h, edgecolor=vs.NEUTRAL["ink"], lw=.8, label=cname)
    ax.set_xticks(x); ax.set_xticklabels(BS.index); ax.set_ylabel("share of total variance"); ax.legend(fontsize=10); vs.panel_label(ax, "a")
    ax = axes[1]; xx = np.arange(len(DS))
    ax.barh(xx + .2, DS["before_share"], .38, color=BEFORE, hatch="//", edgecolor="white", lw=.6, label="before (raw numeric, unscaled)")
    ax.barh(xx - .2, DS["after_share"], .38, color=AFTER, edgecolor="white", lw=.6, label="after (X_main)")
    ax.set_yticks(xx); ax.set_yticklabels(DS.index); ax.set_xlabel("share of mean squared pairwise distance"); ax.legend(); vs.panel_label(ax, "b")
    vs.finalize_figure(fig, FIG, "figD3_encoding_and_distance_share")
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8))
    ax = axes[0]; idx = V.index; xx = np.arange(len(idx))
    ax.bar(xx - .2, V["VIF_with_total"], .38, color=BEFORE, hatch="//", edgecolor="white", lw=.6, label="with total_spend")
    ax.bar(xx + .2, V["VIF_without_total"].fillna(0), .38, color=AFTER, edgecolor="white", lw=.6, label="without total_spend (X_main)")
    ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(idx, rotation=45, ha="right"); ax.set_ylabel("VIF (log scale)"); ax.legend(); vs.panel_label(ax, "a")
    ax = axes[1]
    ax.plot(pc_tab["k"], pc_tab["cum_variance"], marker="o", color=vs.PALETTE["blue"], label="cumulative variance"); ax.plot(pc_tab["k"], pc_tab["nn5_agree"], marker="s", ls="--", color=AFTER, label="5-NN preserved vs full space")
    ax.axhline(.9, color=vs.NEUTRAL["mid"], ls=":"); ax.axvline(pca.n_components_, color=vs.NEUTRAL["mid"], ls=":"); ax.set_xlabel("number of principal components"); ax.legend(); vs.panel_label(ax, "b")
    vs.finalize_figure(fig, FIG, "figD4_redundancy_and_pca")
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8))
    for ax, col, ttl, l in ((axes[0], "nn5_agree_with_main", "5-NN agreement with X_main", "a"), (axes[1], "dist_spearman_with_main", "Spearman of pairwise distances with X_main", "b")):
        v = GEO.set_index("variant")[col]
        ax.bar(range(len(v)), v.values, color=vs.NEUTRAL["light"], edgecolor=vs.NEUTRAL["ink"], lw=.8)
        for b, h in zip(ax.patches, ["", "//", "..", "xx", "\\\\", "", "//"]):
            b.set_hatch(h)
        for xi, vi in enumerate(v.values): ax.text(xi, vi + .01, f"{vi:.2f}", ha="center", fontsize=10)
        ax.set_xticks(range(len(v))); ax.set_xticklabels(v.index, rotation=40, ha="right"); ax.set_ylabel(ttl); ax.set_ylim(0, 1.1); vs.panel_label(ax, l)
    vs.finalize_figure(fig, FIG, "figD5_variants_geometry")
    ck("Đã ghi hình và bảng", len(list(FIG.glob("*.png"))) == 5)

    log.info("-" * 80)
    log.info("KẾT QUẢ NGHIỆM THU: %d/%d đạt", sum(ok for _, ok, _ in ck.rows), len(ck.rows))
    if ck.all_ok:
        files = [OUT / "X_main.csv", OUT / "customer_ids.csv", OUT / "pipeline" / "feature_metadata.csv", OUT / "pipeline" / "prep_config.json", OUT / "variants" / "gower_distance.npy"] + \
                [OUT / "variants" / f"X_{n}.csv" for n in variants_X]
        mp = OUT / "PREPARED_SHA256.json"
        if mp.exists():
            os.chmod(mp, 0o644)
        mp.write_text(json.dumps({str(p.relative_to(ROOT)): sha256(p) for p in files}, indent=1), encoding="utf-8")
        log.info("Đã ghi SHA-256 của %d file đã chuẩn bị vào %s", len(files), mp.relative_to(ROOT))
    return 0 if ck.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
