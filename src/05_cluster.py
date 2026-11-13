#!/usr/bin/env python3
"""Câu e - Phân cụm và đánh giá: dữ liệu trước xử lý (D3, biến đổi tối thiểu) vs sau tiền xử lý + chuẩn bị (câu c, d).

Thang so sánh (cùng 260 dòng ở mọi nhánh chính; số cột khác nhau và được ghi rõ):
  B0  D3 + biến đổi tối thiểu (điền trung vị, one-hot, mã hạng) KHÔNG chuẩn hóa           (20 cột)
  B1  B0 + z-score mọi cột                                                              (20 cột)  <- "Before" chính
  L2  dữ liệu câu c + cùng biến đổi tối thiểu như B1 (tách riêng đóng góp của câu c)    (21 cột, 'unknown' là một mức)
  L3  X_main của câu d (log, z-score, one-hot x w, bỏ total_spend)                      (19 cột)  <- "After" chính
  REF D0 sạch qua đúng pipeline câu d (mốc tham chiếu, không phải một phương pháp)       (19 cột)
Thuật toán: K-Means (chính, n_init = 50); phân cấp liên kết trung bình trên khoảng cách Gower (thuật toán thứ hai, không cần ma trận số).
k chọn CHỈ bằng chỉ số nội tại (Silhouette); nhãn phân khúc ẩn chỉ dùng để đánh giá ngoài (ARI, NMI, purity), không dùng để chọn k, tham số hay biến.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import viz_style as vs  # noqa: E402

ROOT = SRC.parent
DATA, PRE, PREP = ROOT / "data" / "01_generated", ROOT / "data" / "02_preprocessed", ROOT / "data" / "03_prepared"
HID, OUT = ROOT / "data" / "hidden", ROOT / "data" / "04_clustering"
FIG, TAB, LOGD = ROOT / "figures" / "05_clustering", ROOT / "reports" / "tables", ROOT / "logs"
SEED, KS, FIXED_K, N_INIT, B_BOOT = 42, list(range(2, 9)), 4, 50, 1000
SEGS = ["HighValue", "PromoHunter", "Infrequent", "Browser"]
TIERS = {"bronze": 0, "silver": 1, "gold": 2}
GRAY, LGRAY = vs.NEUTRAL["mid"], vs.NEUTRAL["light"]
PIPE_STYLE = {  # màu theo thực thể; không dùng màu đỏ
    "B0": dict(label="B0 raw, minimal, no scaling", color=LGRAY, ls=":", mk="v", hatch="xx"),
    "B1": dict(label="B1 raw, minimal + z-score (Before)", color=vs.PALETTE["violet"], ls="--", mk="D", hatch="//"),
    "L2": dict(label="L2 preprocessed (c), minimal + z-score", color=GRAY, ls="-.", mk="^", hatch=".."),
    "L3": dict(label="L3 prepared X_main (After)", color=vs.PALETTE["teal"], ls="-", mk="o", hatch=""),
    "REF": dict(label="REF clean D0, prepared (ceiling)", color=vs.PALETTE["blue"], ls="-", mk="s", hatch="\\\\")}
log = logging.getLogger("e")


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


def verify_inputs():
    man_c = json.loads((PRE / "FROZEN_SHA256.json").read_text(encoding="utf-8"))
    man_d = json.loads((PREP / "PREPARED_SHA256.json").read_text(encoding="utf-8"))
    bad = [r for r, h in {**man_c["frozen_files"], **man_d}.items() if sha256(ROOT / r) != h]
    if sha256(DATA / "data_raw_corrupted.csv") != man_c["input_raw_corrupted_sha256"]:
        bad.append("data_raw_corrupted.csv")
    if bad:
        raise SystemExit(f"TỪ CHỐI: đầu vào không khớp mã băm đã đóng băng: {bad}")
    return man_c, man_d


# ------------------------------------------------------------------ ma trận
def minimal_matrix(df, scale):
    """Biến đổi tối thiểu để K-Means chạy: điền trung vị cho cột số, mã hạng cho tier, one-hot (NaN -> toàn 0), tùy chọn z-score mọi cột."""
    num = ["age", "n_orders_12m", "avg_order_value", "total_spend_12m", "recency_days", "discount_usage_rate", "sessions_per_month",
           "avg_session_minutes", "pages_per_session", "cart_abandon_rate"]
    X = df[num].astype(float)
    X = X.fillna(X.median())
    X["membership_tier"] = df["membership_tier"].map(TIERS).astype(float)
    X["app_installed"] = df["app_installed"].astype(float)
    X = pd.concat([X, pd.get_dummies(df["device_type"]).astype(float).add_prefix("device_"),
                   pd.get_dummies(df["preferred_category"]).astype(float).add_prefix("cat_")], axis=1)
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns) if scale else X


def gower_generic(df, num, log_cols=()):
    n = len(df)
    D, W = np.zeros((n, n)), np.zeros((n, n))
    cols = {c: (np.log1p(df[c].to_numpy(float)) if c in log_cols else df[c].to_numpy(float)) for c in num}
    cols["membership_tier"] = df["membership_tier"].map(TIERS).to_numpy(float)
    for x in cols.values():
        ok = ~np.isnan(x)
        r = np.nanmax(x) - np.nanmin(x)
        m = ok[:, None] & ok[None, :]
        D += np.where(m, np.abs(np.nan_to_num(x)[:, None] - np.nan_to_num(x)[None, :]) / r, 0); W += m
    for c in ("app_installed", "device_type", "preferred_category"):
        x = df[c].replace("unknown", np.nan).to_numpy(object)
        ok = pd.notna(x)
        m = ok[:, None] & ok[None, :]
        D += np.where(m, (x[:, None] != x[None, :]), 0); W += m
    return D / W


# ------------------------------------------------------------------ đánh giá
def purity(y, lab):
    return float(pd.crosstab(lab, y).max(axis=1).sum() / len(y))


def internal(X, lab):
    return dict(sil=float(silhouette_score(X, lab)), db=float(davies_bouldin_score(X, lab)), ch=float(calinski_harabasz_score(X, lab)))


def kmeans(X, k, seed=SEED, n_init=N_INIT):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


def select_k(X):
    rows = []
    for k in KS:
        km = kmeans(X, k)
        rows.append(dict(k=k, inertia=km.inertia_, **internal(X, km.labels_), min_size=int(np.bincount(km.labels_).min())))
    t = pd.DataFrame(rows)
    return t, int(t.loc[t["sil"].idxmax(), "k"])  # chỉ dùng chỉ số nội tại


def summarize(name, k, lab, X, Xc, y):
    sz = np.bincount(lab)
    return dict(pipeline=name, k=k, n_rows=len(lab), n_cols=X.shape[1], ARI=adjusted_rand_score(y, lab), NMI=normalized_mutual_info_score(y, lab), purity=purity(y, lab),
                **{f"{m}_own": v for m, v in internal(X, lab).items()}, sil_common=float(silhouette_score(Xc, lab)), db_common=float(davies_bouldin_score(Xc, lab)),
                min_cluster=int(sz.min()), max_cluster=int(sz.max()), clusters_under_5pct=int((sz < 0.05 * len(lab)).sum()), sizes=" / ".join(map(str, sorted(sz, reverse=True))))


def stability(X, k, full, reps=30, frac=0.8):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, k]))
    out = []
    for r in range(reps):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        out.append(adjusted_rand_score(full[idx], KMeans(k, n_init=10, random_state=r).fit(X[idx]).labels_))
    runs = [KMeans(k, n_init=1, random_state=100 + r).fit(X) for r in range(20)]
    best = min(runs, key=lambda m: m.inertia_)
    loc = [adjusted_rand_score(best.labels_, m.labels_) for m in runs]
    return float(np.mean(out)), float(np.std(out)), float(np.mean(loc)), float(np.min(loc))


def boot_ci(labs, y, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = [rng.integers(0, n, n) for _ in range(B_BOOT)]
    res = {nm: np.array([adjusted_rand_score(y[i], l[i]) for i in idx]) for nm, l in labs.items()}
    return res


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    LOGD.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOGD / "05_cluster.log", mode="w", encoding="utf-8")])
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    vs.apply_publication_style()
    man_c, man_d = verify_inputs()
    spec = importlib.util.spec_from_file_location("prep04", SRC / "04_prepare.py")
    p4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p4)
    dd = pd.read_csv(DATA / "data_dictionary.csv")
    raw, proc = pd.read_csv(DATA / "data_raw_corrupted.csv"), pd.read_csv(PRE / "data_after_preprocessing.csv")
    flags = pd.read_csv(PRE / "preprocessing_flags.csv")
    Xmain = pd.read_csv(PREP / "X_main.csv")
    truth = pd.read_csv(HID / "ground_truth.csv")           # CHỈ để đánh giá ngoài
    d0 = pd.read_csv(DATA / "data_clean_reference.csv")      # CHỈ để làm mốc tham chiếu
    for d in (proc, flags, truth, d0):
        assert d["customer_id"].equals(raw["customer_id"])
    y = truth["true_segment"].to_numpy()
    cfg = json.loads((PREP / "pipeline" / "prep_config.json").read_text(encoding="utf-8"))
    num_log, num_lin = cfg["log1p_columns"], cfg["linear_columns"]
    _, X0, _ = p4.build(d0, num_log, num_lin); X0.columns = [c.split("__", 1)[1] for c in X0.columns]
    mats = {"B0": minimal_matrix(raw, False), "B1": minimal_matrix(raw, True), "L2": minimal_matrix(proc, True), "L3": Xmain, "REF": X0}
    Xc = Xmain.to_numpy()  # không gian chung để so sánh công bằng các nhãn
    log.info("=" * 84 + "\nCÂU E - PHÂN CỤM. Đầu vào đã đóng băng (mã băm câu c, d khớp). N = 260 ở mọi nhánh chính.\n" + "=" * 84)
    show("0. Không gian đặc trưng của từng nhánh (số hàng x số cột)", pd.DataFrame({k: dict(rows=v.shape[0], cols=v.shape[1], scaled=k != "B0") for k, v in mats.items()}).T)

    # ---------------- chọn k bằng chỉ số nội tại
    sel, kbest, rows_k = {}, {}, []
    for nm, M in mats.items():
        t, kb = select_k(M.to_numpy()); sel[nm], kbest[nm] = t, kb
        rows_k.append(t.assign(pipeline=nm))
    KT = pd.concat(rows_k)
    save_table(KT.round(5), "e_k_selection")
    show("1. Chọn k (Silhouette cao nhất, thang riêng của từng nhánh; DB thấp = tốt, CH cao = tốt)", KT.pivot(index="k", columns="pipeline", values="sil").round(4))
    show("1b. k được chọn bằng Silhouette", pd.Series(kbest, name="k_selected"))
    show("1c. Davies-Bouldin theo k", KT.pivot(index="k", columns="pipeline", values="db").round(3))

    # ---------------- chạy chính: k chọn theo nội tại và k = 4 cố định
    res, labs, centers = [], {}, {}
    for nm, M in mats.items():
        for tag, k in (("ksel", kbest[nm]), ("k4", FIXED_K)):
            km = kmeans(M.to_numpy(), k)
            labs[(nm, tag)] = km.labels_
            centers[(nm, tag)] = pd.DataFrame(km.cluster_centers_, columns=M.columns)
            res.append(dict(setting=tag, **summarize(nm, k, km.labels_, M.to_numpy(), Xc, y)))
    R = pd.DataFrame(res)
    save_table(R.round(5), "e_main_results")
    for tag, title in (("k4", "k = 4 cố định (số phân khúc thiết kế, chỉ dùng để so sánh cùng k)"), ("ksel", "k chọn theo Silhouette của từng nhánh")):
        show(f"2. Kết quả K-Means, {title}", R[R["setting"] == tag].drop(columns="setting").set_index("pipeline").round(4))

    # ---------------- bootstrap khoảng tin cậy cho ARI và chênh lệch
    boots = {}
    for tag in ("k4", "ksel"):
        L = {nm: labs[(nm, tag)] for nm in mats}
        boots[tag] = boot_ci(L, y)
    brow = []
    for tag in ("k4", "ksel"):
        b = boots[tag]
        for nm in mats:
            lo, hi = ci(b[nm]); brow.append(dict(setting=tag, comparison=nm, ARI=float(adjusted_rand_score(y, labs[(nm, tag)])), ci_low=lo, ci_high=hi))
        for a, c in (("L3", "B1"), ("L3", "L2"), ("L2", "B1"), ("B1", "B0"), ("REF", "L3")):
            d = b[a] - b[c]; lo, hi = ci(d)
            brow.append(dict(setting=tag, comparison=f"{a} - {c}", ARI=float(adjusted_rand_score(y, labs[(a, tag)]) - adjusted_rand_score(y, labs[(c, tag)])), ci_low=lo, ci_high=hi,
                             frac_boot_positive=float((d > 0).mean())))
    BT = pd.DataFrame(brow)
    save_table(BT.round(5), "e_bootstrap_ari")
    show("3. ARI và chênh lệch ARI với khoảng tin cậy 95% (bootstrap 1000 lần theo dòng, ghép cặp)", BT.round(4))

    # ---------------- ổn định
    srow = []
    for nm, M in mats.items():
        for tag in ("k4", "ksel"):
            k = FIXED_K if tag == "k4" else kbest[nm]
            m, s, lm, lmin = stability(M.to_numpy(), k, labs[(nm, tag)])
            srow.append(dict(pipeline=nm, setting=tag, k=k, subsample80_ARI_mean=m, subsample80_ARI_sd=s, single_init_ARI_mean=lm, single_init_ARI_min=lmin))
    ST = pd.DataFrame(srow)
    save_table(ST.round(5), "e_stability")
    show("4. Độ ổn định: ARI giữa phân cụm trên mẫu con 80% (30 lần) và phân cụm đầy đủ; ARI giữa 20 lần chạy n_init = 1 và lần tốt nhất", ST.round(4))

    # ---------------- tác động của dòng hỏng lên cụm (K-Means trước xử lý)
    rep = flags["n_cells_invalid_repaired"].to_numpy().astype(bool)
    crow = []
    for nm in ("B0", "B1", "L2", "L3"):
        for tag in ("k4", "ksel"):
            lab = labs[(nm, tag)]
            sz = np.bincount(lab)
            dom = [int(c) for c in range(len(sz)) if sz[c] <= 10 and rep[lab == c].sum() >= max(1, 0.5 * sz[c])]
            crow.append(dict(pipeline=nm, setting=tag, k=len(sz), n_repaired_rows=int(rep.sum()), n_clusters_size_le_10=int((sz <= 10).sum()),
                             n_small_clusters_mostly_repaired_rows=len(dom), rows_of_repaired_in_smallest_cluster=int(rep[lab == int(np.argmin(sz))].sum()), smallest_cluster=int(sz.min())))
    CR = pd.DataFrame(crow)
    save_table(CR, "e_corrupted_rows_in_clusters")
    show("5. 7 dòng có ô hỏng nằm ở đâu trong các cụm", CR)

    vlabs = {}
    # ---------------- thuật toán thứ hai: phân cấp liên kết trung bình trên Gower
    G = {"B1_gower_raw": gower_generic(raw, [c for c in ["age", "n_orders_12m", "avg_order_value", "total_spend_12m", "recency_days", "discount_usage_rate", "sessions_per_month",
                                                         "avg_session_minutes", "pages_per_session", "cart_abandon_rate"]]),
         "L2_gower_processed_raw_units": gower_generic(proc, ["age", "n_orders_12m", "avg_order_value", "total_spend_12m", "recency_days", "discount_usage_rate", "sessions_per_month",
                                                              "avg_session_minutes", "pages_per_session", "cart_abandon_rate"]),
         "L3_gower_prepared": p4.gower(proc, num_log, num_lin)}
    grow, glabs = [], {}
    for nm, Gm in G.items():
        for k in sorted({FIXED_K, kbest["L3"]}):
            for link in ("average", "complete"):
                lab = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=link).fit_predict(Gm)
                glabs[(nm, k, link)] = lab
                sz = np.bincount(lab)
                grow.append(dict(pipeline=nm, k=k, linkage=link, ARI=adjusted_rand_score(y, lab), NMI=normalized_mutual_info_score(y, lab), purity=purity(y, lab),
                                 sil_gower=float(silhouette_score(Gm, lab, metric="precomputed")) if len(set(lab)) > 1 else np.nan, sil_common=float(silhouette_score(Xc, lab)),
                                 min_cluster=int(sz.min()), sizes=" / ".join(map(str, sorted(sz, reverse=True)))))
    GR = pd.DataFrame(grow)
    GR["selected_by_gower_silhouette"] = GR.groupby(["pipeline", "k"])["sil_gower"].transform(lambda s: s == s.max())
    save_table(GR.round(5), "e_gower_hierarchical")
    show("6. Thuật toán thứ hai: phân cấp trên khoảng cách Gower. Liên kết chọn trước = average; thêm complete và chọn giữa hai loại CHỈ bằng Silhouette Gower (không dùng nhãn); in cả hai", GR.round(4))

    # ---------------- độ nhạy (dựa trên lựa chọn ở câu c, d), cùng k như L3
    vrows = []
    vdir = PREP / "variants"
    def run_variant(name, X, k, rows=None):
        Xv = X.to_numpy() if hasattr(X, "to_numpy") else X
        km = kmeans(Xv, k)
        idx = np.arange(len(Xv)) if rows is None else rows
        base = labs[("L3", "k4" if k == FIXED_K else "ksel")]
        vlabs[(name, k)] = km.labels_
        return dict(variant=name, k=k, n_rows=len(Xv), n_cols=Xv.shape[1], ARI_vs_truth=adjusted_rand_score(y[idx], km.labels_), NMI_vs_truth=normalized_mutual_info_score(y[idx], km.labels_),
                    ARI_vs_main_labels=adjusted_rand_score(base[idx], km.labels_), sil_own=float(silhouette_score(Xv, km.labels_))), km.labels_
    for k in sorted({FIXED_K, kbest["L3"]}):
        main_r, _ = run_variant("main X_main", Xmain, k); vrows.append(main_r)
        for nm in ("full_features", "cat_unknown_col", "cat_knn7", "cat_mode", "no_category", "pca90"):
            r, l = run_variant(nm, pd.read_csv(vdir / f"X_{nm}.csv"), k); vrows.append(r); vlabs[(nm, k)] = l
        keep = np.flatnonzero(flags["extreme_tukey3_valid"].to_numpy() == 0)
        r, _ = run_variant(f"main without {int((flags['extreme_tukey3_valid'] == 1).sum())} valid extreme rows (N changes)", Xmain.iloc[keep], k, rows=keep); vrows.append(r)
        # độ phủ cùng tập dòng: nhãn của lần chạy đầy đủ cắt về tập dòng đó
        full = labs[("L3", "k4" if k == FIXED_K else "ksel")][keep]
        vrows[-1]["ARI_full_run_labels_on_same_rows_vs_truth"] = adjusted_rand_score(y[keep], full)
    VR = pd.DataFrame(vrows)
    bv = boot_ci({n: vlabs[(n, FIXED_K)] for n in ("main X_main", "full_features", "cat_unknown_col", "cat_knn7", "cat_mode", "no_category", "pca90")}, y)
    for i, r in VR.iterrows():
        if r["k"] == FIXED_K and r["variant"] in bv and r["variant"] != "main X_main":
            d = bv[r["variant"]] - bv["main X_main"]; lo, hi = ci(d)
            VR.loc[i, ["dARI_vs_main", "dARI_ci_low", "dARI_ci_high"]] = [float(r["ARI_vs_truth"] - VR.loc[(VR["variant"] == "main X_main") & (VR["k"] == FIXED_K), "ARI_vs_truth"].iloc[0]), lo, hi]
    save_table(VR.round(5), "e_sensitivity")
    show("7. Độ nhạy: biến thể từ câu c, d (cùng k; 'ARI_vs_main_labels' đo mức đồng thuận với phân cụm chính)", VR.round(4))

    # ---------------- ceiling theo khách "cầu nối"
    bridge = truth["is_bridge"].to_numpy().astype(bool)
    brow2 = []
    for nm in mats:
        lab = labs[(nm, "k4")]
        brow2.append(dict(pipeline=nm, ARI_non_bridge=adjusted_rand_score(y[~bridge], lab[~bridge]), ARI_bridge=adjusted_rand_score(y[bridge], lab[bridge]), n_non_bridge=int((~bridge).sum()), n_bridge=int(bridge.sum())))
    BR = pd.DataFrame(brow2)
    save_table(BR.round(5), "e_bridge_customers")
    show("8. ARI tách theo khách thường / khách 'cầu nối' (k = 4)", BR.round(4))

    # ---------------- đặc điểm cụm (k = 4), tính trên đơn vị gốc của dữ liệu đã xử lý để cả hai nhánh cùng thang
    numc = ["age", "n_orders_12m", "avg_order_value", "total_spend_12m", "recency_days", "discount_usage_rate", "sessions_per_month", "avg_session_minutes", "pages_per_session", "cart_abandon_rate"]
    prof, ctab = {}, {}
    for nm in ("B1", "L3"):
        lab = labs[(nm, "k4")]
        P = proc.assign(cluster=lab).groupby("cluster")[numc].median()
        P["size"] = np.bincount(lab)
        for c in ("membership_tier", "device_type", "preferred_category"):
            P[c + "_mode"] = proc.assign(cluster=lab).groupby("cluster")[c].agg(lambda s: s.value_counts().index[0])
        P["app_installed_share"] = proc.assign(cluster=lab).groupby("cluster")["app_installed"].mean()
        prof[nm] = P
        ctab[nm] = pd.crosstab(pd.Series(lab, name="cluster"), pd.Series(y, name="hidden_segment")).reindex(columns=SEGS, fill_value=0)
        save_table(P.round(3), f"e_cluster_profile_{nm}_k4", index=True)
        save_table(ctab[nm], f"e_contingency_{nm}_k4", index=True)
        show(f"9. Hồ sơ cụm {nm}, k = 4 (trung vị theo đơn vị gốc của dữ liệu đã xử lý)", P.round(2))
        show(f"9b. Cụm {nm} x phân khúc ẩn (chỉ để đánh giá)", ctab[nm])

    # ---------------- lưu nhãn, tâm cụm, cấu hình
    L = pd.DataFrame({"customer_id": raw["customer_id"]})
    for (nm, tag), lab in labs.items():
        L[f"{nm}_kmeans_{tag}"] = lab
    for (nm, k, link), lab in glabs.items():
        L[f"{nm}_hier_{link}_k{k}"] = lab
    L.to_csv(OUT / "labels.csv", index=False)
    for (nm, tag), c in centers.items():
        c.round(6).to_csv(OUT / f"centers_{nm}_{tag}.csv", index_label="cluster")
    run_cfg = dict(seed=SEED, k_grid=KS, fixed_k=FIXED_K, n_init=N_INIT, bootstrap=B_BOOT, k_selected_by_silhouette=kbest, versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
                   inputs_sha256={**man_c["frozen_files"], **man_d, "data_raw_corrupted.csv": man_c["input_raw_corrupted_sha256"]},
                   columns={k: int(v.shape[1]) for k, v in mats.items()}, hidden_labels_used_for="đánh giá ngoài và mốc tham chiếu REF; không dùng để chọn k, tham số hay biến")
    (OUT / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # ---------------- hình
    def sty(nm): return PIPE_STYLE[nm]
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8))
    for ax, (col, ttl, l) in zip(axes, (("sil", "Silhouette (own space)", "a"), ("db", "Davies-Bouldin (lower = better)", "b"), ("ch", "Calinski-Harabasz", "c"))):
        for nm, t in sel.items():
            ax.plot(t["k"], t[col], color=sty(nm)["color"], ls=sty(nm)["ls"], marker=sty(nm)["mk"], label=sty(nm)["label"])
        ax.set_xlabel("k"); ax.set_ylabel(ttl); vs.panel_label(ax, l)
    axes[0].legend(fontsize=9)
    vs.finalize_figure(fig, FIG, "figE1_k_selection")

    def bars(ax, tag, col, ttl, with_ci=False):
        sub = R[R["setting"] == tag].set_index("pipeline").loc[list(mats)]
        for i, nm in enumerate(mats):
            v = sub.at[nm, col]
            err = None
            if with_ci and col == "ARI":
                lo, hi = ci(boots[tag][nm]); err = [[v - lo], [hi - v]]
            ax.bar(i, v, .7, color=sty(nm)["color"], hatch=sty(nm)["hatch"], edgecolor=vs.NEUTRAL["ink"], lw=.8, yerr=err, capsize=4)
            ax.text(i, (hi if err else v) + .015, f"{v:.2f}", ha="center", fontsize=10)
        ax.set_xticks(range(len(mats))); ax.set_xticklabels(list(mats)); ax.set_ylabel(ttl); ax.set_ylim(0, 1.05)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    bars(axes[0, 0], "k4", "ARI", "ARI vs hidden segments (k = 4, 95% bootstrap CI)", True); vs.panel_label(axes[0, 0], "a")
    bars(axes[0, 1], "k4", "NMI", "NMI vs hidden segments (k = 4)"); vs.panel_label(axes[0, 1], "b")
    bars(axes[1, 0], "ksel", "ARI", "ARI (k selected by Silhouette, 95% bootstrap CI)", True); vs.panel_label(axes[1, 0], "c")
    axes[1, 0].set_xticklabels([f"{nm}\nk={kbest[nm]}" for nm in mats]); axes[1, 1].set_xticklabels([f"{nm}\nk={kbest[nm]}" for nm in mats])
    bars(axes[1, 1], "ksel", "NMI", "NMI (k selected by Silhouette)"); vs.panel_label(axes[1, 1], "d")
    axes[1, 1].set_xticklabels([f"{nm}\nk={kbest[nm]}" for nm in mats])
    axes[0, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=sty(n)["color"], hatch=sty(n)["hatch"], edgecolor=vs.NEUTRAL["ink"], label=sty(n)["label"]) for n in mats], fontsize=8, loc="upper left")
    vs.finalize_figure(fig, FIG, "figE2_external_metrics")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    s4 = ST[ST["setting"] == "k4"].set_index("pipeline").loc[list(mats)]
    for i, nm in enumerate(mats):
        axes[0].bar(i, s4.at[nm, "subsample80_ARI_mean"], .7, yerr=s4.at[nm, "subsample80_ARI_sd"], capsize=4, color=sty(nm)["color"], hatch=sty(nm)["hatch"], edgecolor=vs.NEUTRAL["ink"], lw=.8)
        axes[1].bar(i, s4.at[nm, "single_init_ARI_mean"], .7, color=sty(nm)["color"], hatch=sty(nm)["hatch"], edgecolor=vs.NEUTRAL["ink"], lw=.8)
    for ax, ttl, l in ((axes[0], "ARI: 80% subsample vs full run (mean ± sd, k = 4)", "a"), (axes[1], "ARI: single-init runs vs best run (k = 4)", "b")):
        ax.set_xticks(range(len(mats))); ax.set_xticklabels(list(mats)); ax.set_ylabel(ttl); ax.set_ylim(0, 1.05); vs.panel_label(ax, l)
    for off, nm in zip((-.2, .2), ("B1", "L3")):
        sz = sorted(np.bincount(labs[(nm, "k4")]), reverse=True)
        axes[2].bar(np.arange(4) + off, sz, .38, color=sty(nm)["color"], hatch=sty(nm)["hatch"], edgecolor="white", lw=.6, label=sty(nm)["label"])
        for xi, v in zip(np.arange(4) + off, sz): axes[2].text(xi, v + 2, str(v), ha="center", fontsize=10)
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels([f"cluster #{i + 1}\n(sorted by size)" for i in range(4)]); axes[2].set_ylabel("rows (k = 4)"); axes[2].legend(fontsize=9); vs.panel_label(axes[2], "c")
    vs.finalize_figure(fig, FIG, "figE3_stability_and_sizes")

    P2 = PCA(2, random_state=0).fit(Xc); Z = P2.transform(Xc)
    fig, axes = plt.subplots(1, 3, figsize=(23, 7))
    mk = ["o", "s", "^", "D", "v", "P", "X"]
    for ax, (ttl, lab, names, l) in zip(axes, (("Hidden segments (diagnostic only)", y, SEGS, "a"), ("Before: B1 K-Means, k = 4", labs[("B1", "k4")], None, "b"), ("After: L3 K-Means, k = 4", labs[("L3", "k4")], None, "c"))):
        cats = names or sorted(set(lab))
        for i, cval in enumerate(cats):
            m = (lab == cval)
            ax.scatter(Z[m, 0], Z[m, 1], s=34, marker=mk[i], color=vs.SERIES[i % 4], alpha=.8, edgecolor="none", label=str(cval) if names else f"cluster {cval + 1} (n={int(m.sum())})")
        ax.set_title(ttl, fontsize=12); ax.set_xlabel(f"PC1 of X_main ({100 * P2.explained_variance_ratio_[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({100 * P2.explained_variance_ratio_[1]:.0f}%)")
        ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, columnspacing=0.8, handletextpad=0.3); vs.panel_label(ax, l)
    vs.finalize_figure(fig, FIG, "figE4_pca_embedding")

    fig, axes = plt.subplots(1, 3, figsize=(24, 5.6), gridspec_kw={"width_ratios": [1, 1, 0.03]})
    for ax, nm, ttl, l in ((axes[0], "B1", "Before (B1) clusters, mean of X_main columns", "a"), (axes[1], "L3", "After (L3) clusters, mean of X_main columns", "b")):
        M = Xmain.assign(cluster=labs[(nm, "k4")]).groupby("cluster").mean()
        im = ax.imshow(M.to_numpy(), cmap=vs.CMAP_DIV, vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=60, ha="right", fontsize=9); ax.set_yticks(range(4)); ax.set_yticklabels([f"cluster {i + 1} (n={n})" for i, n in enumerate(np.bincount(labs[(nm, 'k4')]))])
        ax.set_title(ttl, fontsize=12); vs.panel_label(ax, l)
        for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im, cax=axes[2], label="mean of standardized feature")
    vs.finalize_figure(fig, FIG, "figE5_cluster_profiles_common_space", pad=1.5)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.4), gridspec_kw={"width_ratios": [1, 1, 0.04]})
    for ax, nm, ttl, l in ((axes[0], "B1", "Before (B1), k = 4", "a"), (axes[1], "L3", "After (L3), k = 4", "b")):
        T = ctab[nm]; im = ax.imshow(T.to_numpy(), cmap=vs.CMAP_SEQ, vmin=0, vmax=max(ctab["B1"].to_numpy().max(), ctab["L3"].to_numpy().max()))
        for i in range(T.shape[0]):
            for j in range(T.shape[1]): ax.text(j, i, str(T.iat[i, j]), ha="center", va="center", color="white" if T.iat[i, j] > 35 else vs.NEUTRAL["ink"], fontsize=12)
        ax.set_xticks(range(4)); ax.set_xticklabels(SEGS, rotation=20); ax.set_yticks(range(4)); ax.set_yticklabels([f"cluster {i + 1}" for i in range(T.shape[0])]); ax.set_title(ttl, fontsize=12); vs.panel_label(ax, l)
        for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im, cax=axes[2], label="rows")
    vs.finalize_figure(fig, FIG, "figE6_cluster_vs_hidden_segment", pad=1.5)

    fig, ax = plt.subplots(figsize=(14, 6.4))
    sub = VR[VR["k"] == FIXED_K].reset_index(drop=True)
    ax.barh(range(len(sub)), sub["ARI_vs_truth"], color=[vs.PALETTE["teal"] if v.startswith("main X") else LGRAY for v in sub["variant"]], edgecolor=vs.NEUTRAL["ink"], lw=.8)
    for i, (a, b) in enumerate(zip(sub["ARI_vs_truth"], sub["ARI_vs_main_labels"])): ax.text(a + .01, i, f"ARI {a:.2f} | agreement with main {b:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels([f"{v}  [N={n}, cols={c}]" for v, n, c in zip(sub["variant"], sub["n_rows"], sub["n_cols"])], fontsize=9); ax.invert_yaxis(); ax.set_xlabel("ARI vs hidden segments (k = 4)"); ax.set_xlim(0, 1.15)
    vs.finalize_figure(fig, FIG, "figE7_sensitivity")
    files = [OUT / "labels.csv", OUT / "run_config.json"] + sorted(OUT.glob("centers_*.csv"))
    (OUT / "RESULTS_SHA256.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha256(p) for p in files}, indent=1), encoding="utf-8")
    log.info("\nĐã ghi %d hình vào %s; nhãn vào %s; mã băm %d file kết quả", len(list(FIG.glob("*.png"))), FIG, OUT / "labels.csv", len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
