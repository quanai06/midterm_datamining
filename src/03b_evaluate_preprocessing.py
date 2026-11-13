#!/usr/bin/env python3
"""Câu c - Đối chiếu với dữ liệu sạch tham chiếu, CHẠY SAU KHI ĐÃ ĐÓNG BĂNG kết quả.

Điều kiện chạy: SHA-256 của dữ liệu xử lý, nhật ký, cờ, biến thể và D3 phải khớp data/02_preprocessed/FROZEN_SHA256.json; nếu không, script từ chối.
Script chỉ ĐO và không ghi vào bất kỳ file dữ liệu nào ở data/. Dữ liệu sạch tham chiếu (D0) chỉ dùng để tính số đo tổng hợp; không dùng để thay giá trị.

Phân biệt rõ hai loại số đo (không gọi lẫn nhau):
  (A) Sai khác phân phối của cả cột so với D0 (KS, Wasserstein, trung vị, độ lệch...). Không phải sai số từng ô.
  (B) Sai số tổng hợp tại các ô đã bị thay đổi (MAE, RMSE, trung vị sai số tuyệt đối, độ chệch), so với D0 và so với mốc trung vị toàn cục.
      Lưu ý D0 là giá trị trước khi thêm nhiễu: ngay cả giá trị quan sát đúng (D2) cũng lệch D0 một lượng "sàn nhiễu" được báo cáo riêng.
"""
from __future__ import annotations

import hashlib
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
from scipy import stats
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_style as vs  # noqa: E402
from dm_utils import hopkins, knn_sets, make_lens, select_log_cols  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, HID, OUT = ROOT / "data" / "01_generated", ROOT / "data" / "hidden", ROOT / "data" / "02_preprocessed"
FIG, TAB, LOGD = ROOT / "figures" / "03_preprocessing_eval", ROOT / "reports" / "tables", ROOT / "logs"
SEED = 42
CLEAN, BEFORE, AFTER = vs.STAGE_COLORS["D0 clean"], vs.PALETTE["violet"], vs.PALETTE["teal"]
LS = {"D0 clean": "-", "Before (D3)": ":", "After": "--"}
log = logging.getLogger("c_eval")


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


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def verify_freeze():
    man = json.loads((OUT / "FROZEN_SHA256.json").read_text(encoding="utf-8"))
    bad = [rel for rel, h in man["frozen_files"].items() if sha256(ROOT / rel) != h]
    if sha256(DATA / "data_raw_corrupted.csv") != man["input_raw_corrupted_sha256"]:
        bad.append("data/01_generated/data_raw_corrupted.csv")
    if sha256(ROOT / "src" / "03_preprocess.py") != man["script_sha256"]:
        bad.append("src/03_preprocess.py (script đã đổi sau khi đóng băng)")
    if bad:
        raise SystemExit(f"TỪ CHỐI: mã băm không khớp bản đã đóng băng: {bad}. Chạy lại 03_preprocess.py và nghiệm thu trước.")
    return man


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOGD / "03b_evaluate_preprocessing.log", mode="w", encoding="utf-8")])
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    vs.apply_publication_style()
    man = verify_freeze()
    log.info("=" * 80 + "\nĐỐI CHIẾU SAU ĐÓNG BĂNG: mã băm khớp FROZEN_SHA256.json (%d file + D3 + script)\n" % len(man["frozen_files"]) + "=" * 80)
    dd = pd.read_csv(DATA / "data_dictionary.csv")
    num = dd.loc[dd["dtype"].isin(["int", "float"]), "attribute"].tolist()
    d3, proc = pd.read_csv(DATA / "data_raw_corrupted.csv"), pd.read_csv(OUT / "data_after_preprocessing.csv")
    au = pd.read_csv(OUT / "preprocessing_audit.csv")
    var = pd.read_csv(OUT / "variants" / "preferred_category_variants.csv")
    d0, d2 = pd.read_csv(DATA / "data_clean_reference.csv"), pd.read_csv(DATA / "data_after_outliers.csv")
    truth = pd.read_csv(HID / "ground_truth.csv")
    for d in (d0, d2, proc, truth, var):
        assert d["customer_id"].equals(d3["customer_id"]), "thứ tự customer_id không khớp"
    log_cols, _ = select_log_cols(d3, dd, num)  # cùng quy tắc suy từ D3 như ở câu b
    idx = {cid: i for i, cid in enumerate(d3["customer_id"])}
    stages = {"D0 clean": d0, "Before (D3)": d3, "After": proc}

    # ------------------------------------------------------------------ (A) sai khác phân phối của cả cột
    rows = []
    for c in num:
        r0 = d0[c].astype(float)
        for nm, d in stages.items():
            x = d[c].astype(float).dropna()
            row = dict(column=c, stage=nm, n=len(x), median=x.median(), mean=x.mean(), std=x.std(), skew=x.skew(), max=x.max())
            if nm != "D0 clean":
                row["KS_vs_D0"] = stats.ks_2samp(r0, x).statistic
                row["Wasserstein_vs_D0"] = stats.wasserstein_distance(r0, x)
            rows.append(row)
    A = pd.DataFrame(rows)
    show("A. Sai khác phân phối CẢ CỘT so với D0 (không phải sai số từng ô)", A.round(3))
    save_table(A.round(5), "c_eval_distribution_vs_clean")
    piv = A.pivot(index="column", columns="stage")
    summ = pd.DataFrame({"KS_before": piv[("KS_vs_D0", "Before (D3)")], "KS_after": piv[("KS_vs_D0", "After")],
                         "W1_before": piv[("Wasserstein_vs_D0", "Before (D3)")], "W1_after": piv[("Wasserstein_vs_D0", "After")],
                         "std_ratio_before": piv[("std", "Before (D3)")] / piv[("std", "D0 clean")], "std_ratio_after": piv[("std", "After")] / piv[("std", "D0 clean")],
                         "skew_D0": piv[("skew", "D0 clean")], "skew_before": piv[("skew", "Before (D3)")], "skew_after": piv[("skew", "After")]})
    show("A. Tóm tắt theo cột (std_ratio = std / std của D0; 1,0 là khớp)", summ.round(3))

    # ------------------------------------------------------------------ (B) sai số tổng hợp tại các ô đã thay đổi
    nau = au[au["column"].isin(num)].copy()
    nau["after"] = pd.to_numeric(nau["value_after"])
    nau["before"] = pd.to_numeric(nau["value_before"])
    nau["i"] = nau["customer_id"].map(idx)
    nau["d0"] = [d0.at[i, c] for i, c in zip(nau["i"], nau["column"])]
    nau["d2"] = [d2.at[i, c] for i, c in zip(nau["i"], nau["column"])]
    invalid_keys = set(zip(nau.loc[nau["action"] == "invalid_cell->imputed", "customer_id"], nau.loc[nau["action"] == "invalid_cell->imputed", "column"]))
    base_med = {}
    for c in nau["column"].unique():
        x = d3[c].astype(float).copy()
        for cid, cc in invalid_keys:
            if cc == c:
                x[idx[cid]] = np.nan
        base_med[c] = float(x.median())
    nau["baseline_median"] = nau["column"].map(base_med)
    sd0 = d0[num].astype(float).std()
    out = []
    for (c, a), g in nau.groupby(["column", "action"]):
        e, eb = g["after"] - g["d0"], g["baseline_median"] - g["d0"]
        r = dict(column=c, action=a, n_cells=len(g), MAE_after_vs_D0=e.abs().mean(), RMSE_after_vs_D0=float(np.sqrt((e ** 2).mean())), MedAE_after_vs_D0=e.abs().median(),
                 bias_after_minus_D0=e.mean(), MAE_global_median_vs_D0=eb.abs().mean(), MAE_after_relative_to_sd_D0=e.abs().mean() / sd0[c],
                 MAE_global_median_relative_to_sd_D0=eb.abs().mean() / sd0[c], noise_floor_MAE_D2_vs_D0=(g["d2"] - g["d0"]).abs().mean(),
                 MAE_after_vs_D2=(g["after"] - g["d2"]).abs().mean(), std_after_over_std_D0=g["after"].std() / g["d0"].std() if len(g) > 2 else np.nan,
                 corr_after_D0=g["after"].corr(g["d0"]) if len(g) > 4 else np.nan)
        if a == "invalid_cell->imputed":
            r["MAE_before_vs_D0"] = (g["before"] - g["d0"]).abs().mean()
        out.append(r)
    B = pd.DataFrame(out)
    show("B. Sai số TỔNG HỢP tại các ô đã thay đổi, đo bằng D0 (sd0 = độ lệch chuẩn cả cột D0). noise_floor = |D2 - D0| trên cùng các ô", B.round(3))
    save_table(B.round(5), "c_eval_cell_error_aggregate")
    det = nau[nau["action"] == "invalid_cell->imputed"][["customer_id", "column", "before", "after", "d0", "baseline_median"]]
    show("B. Chi tiết 7 ô không hợp lệ (mô tả, n = 1 mỗi cột nên không có ý nghĩa thống kê)", det.round(3))
    save_table(det.round(4), "c_eval_invalid_cells_detail")

    # ------------------------------------------------------------------ (C) phục hồi cấu trúc
    lens = {nm: {k: make_lens(d, num, k, log_cols) for k in ("raw_z", "robust")} for nm, d in stages.items()}
    seg = truth["true_segment"].to_numpy()
    P0, S0 = d0[num].astype(float).corr(), d0[num].astype(float).corr(method="spearman")
    iu = np.triu_indices(len(num), 1)
    rows = []
    for nm in ("D0 clean", "Before (D3)", "After"):
        X = stages[nm][num].astype(float)
        dP, dS = (X.corr() - P0).to_numpy()[iu], (X.corr(method="spearman") - S0).to_numpy()[iu]
        for k in ("raw_z", "robust"):
            n0, nk = knn_sets(lens["D0 clean"][k]), knn_sets(lens[nm][k])
            rows.append(dict(stage=nm, lens=k, mean_abs_dPearson=np.abs(dP).mean(), max_abs_dPearson=np.abs(dP).max(), mean_abs_dSpearman=np.abs(dS).mean(),
                             max_abs_dSpearman=np.abs(dS).max(), nn5_preserved=float(np.mean([len(a & b) / 5 for a, b in zip(n0, nk)])),
                             dist_spearman=float(stats.spearmanr(pdist(lens["D0 clean"][k]), pdist(lens[nm][k]))[0]), hopkins=hopkins(lens[nm][k], SEED)[0],
                             silhouette_hidden_labels=float(silhouette_score(lens[nm][k], seg))))
    C = pd.DataFrame(rows)
    show("C. Cấu trúc so với D0 (silhouette theo nhãn ẩn chỉ để chẩn đoán độ tách, không phải kết quả phân cụm)", C.round(4))
    save_table(C.round(5), "c_eval_structure_vs_clean")

    # ------------------------------------------------------------------ (D) preferred_category
    miss = var.index[var["knn7_max_prob"].notna()]
    t0 = d0.loc[miss, "preferred_category"]
    cat_rows = []
    for nm, col in (("knn7", "preferred_category_knn7"), ("global_mode", "preferred_category_mode")):
        k = int((var.loc[miss, col].to_numpy() == t0.to_numpy()).sum())
        lo, hi = wilson(k, len(miss))
        cat_rows.append(dict(variant=nm, n_cells=len(miss), n_correct=k, accuracy=k / len(miss), wilson95_low=lo, wilson95_high=hi))
    D = pd.DataFrame(cat_rows)
    show("D. preferred_category: 13 ô thiếu, nhãn biến thể so với D0 (n quá nhỏ, khoảng tin cậy rộng)", D.round(3))
    save_table(D.round(4), "c_eval_category")
    props = pd.DataFrame({"D0": d0["preferred_category"].value_counts(), "main (unknown)": var["preferred_category_main"].value_counts(),
                          "knn7": var["preferred_category_knn7"].value_counts(), "mode": var["preferred_category_mode"].value_counts()}).fillna(0).astype(int)
    show("D. Phân bố nhãn theo từng biến thể", props)

    # ------------------------------------------------------------------ hình
    changed_cols = [c for c in num if (proc[c] != d3[c]).any()]
    fig, axes = plt.subplots(2, 4, figsize=(23, 10))
    for ax, c, l in zip(axes.ravel(), changed_cols, "abcdefg"):
        for nm, d in stages.items():
            x = np.sort(d[c].dropna().astype(float).to_numpy())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color={"D0 clean": CLEAN, "Before (D3)": BEFORE, "After": AFTER}[nm], ls=LS[nm], label=nm)
        if c in ("avg_order_value", "sessions_per_month", "avg_session_minutes"):
            ax.set_xscale("log")
        ax.set_title(f"{c}\nKS before {summ.at[c, 'KS_before']:.3f} -> after {summ.at[c, 'KS_after']:.3f}", fontsize=11); ax.set_ylabel("ECDF" if l in "ae" else ""); vs.panel_label(ax, l)
    axes[0, 0].legend(loc="lower right")
    for ax in axes.ravel()[len(changed_cols):]:
        ax.axis("off")
    vs.finalize_figure(fig, FIG, "figE1_ecdf_vs_clean")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))
    for ax, c, unit, l in ((axes[0], "age", "years", "a"), (axes[1], "avg_session_minutes", "min", "b")):
        g = nau[(nau["column"] == c) & (nau["action"] == "missing->imputed")]
        data = [(g["after"] - g["d0"]).abs(), (g["baseline_median"] - g["d0"]).abs(), (g["d2"] - g["d0"]).abs()]
        bp = ax.boxplot(data, patch_artist=True, widths=.55, medianprops=dict(color=vs.NEUTRAL["ink"], lw=2), flierprops=dict(marker="o", markersize=4, markeredgecolor="none", markerfacecolor=vs.NEUTRAL["mid"]))
        for patch, col, h in zip(bp["boxes"], (AFTER, vs.NEUTRAL["light"], CLEAN), ("", "//", "..")):
            patch.set_facecolor(col); patch.set_hatch(h); patch.set_edgecolor(vs.NEUTRAL["ink"])
        ax.set_xticklabels(["After\n(selected)", "Global median\nbaseline", "Noise floor\nabs(D2 - D0)" + ("\n(age: no noise added)" if c == "age" else "")]); ax.set_ylabel(f"absolute error vs D0 ({unit})"); ax.set_title(f"{c}, {len(g)} imputed cells", fontsize=12); vs.panel_label(ax, l)
    ax = axes[2]
    Bm = B[B["action"] == "missing->imputed"].set_index("column")
    Bi = B[B["action"] == "invalid_cell->imputed"].set_index("column")
    allc = list(Bi.index)
    x = np.arange(len(allc))
    ax.bar(x - .2, [Bi.at[c, "MAE_after_relative_to_sd_D0"] for c in allc], .38, color=AFTER, edgecolor="white", lw=.6, label="After (selected)")
    ax.bar(x + .2, [Bi.at[c, "MAE_global_median_relative_to_sd_D0"] for c in allc], .38, color=vs.NEUTRAL["light"], hatch="//", edgecolor=vs.NEUTRAL["ink"], lw=.6, label="Global median")
    ax.set_xticks(x); ax.set_xticklabels(allc, rotation=40, ha="right"); ax.set_ylabel("error / sd of D0 column"); ax.set_title("the 7 invalid cells (n = 1 per column, descriptive)", fontsize=12); ax.legend(); vs.panel_label(ax, "c")
    vs.finalize_figure(fig, FIG, "figE2_cell_error_aggregate")

    fig, axes = plt.subplots(2, 2, figsize=(17, 11.5))
    Cp = C[C["stage"] != "D0 clean"]
    for ax, cols, ttl, l in ((axes[0, 0], ["mean_abs_dPearson", "mean_abs_dSpearman"], "mean absolute correlation difference vs D0", "a"), (axes[0, 1], ["nn5_preserved", "dist_spearman"], "neighbourhood / distance agreement with D0", "b")):
        x = np.arange(len(cols))
        for k, (nm, col, h) in enumerate((("Before (D3)", BEFORE, "//"), ("After", AFTER, ""))):
            sub = Cp[(Cp["stage"] == nm) & (Cp["lens"] == "robust")] if ax is axes[0, 1] else Cp[(Cp["stage"] == nm) & (Cp["lens"] == "raw_z")].iloc[:1]
            v = [sub[c].iloc[0] for c in cols]
            ax.bar(x + (k - .5) * .38, v, .36, color=col, hatch=h, edgecolor="white", lw=.6, label=nm)
            for xi, vi in zip(x + (k - .5) * .38, v):
                ax.text(xi, vi + .005, f"{vi:.3f}", ha="center", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(cols); ax.set_ylabel(ttl); ax.legend(); vs.panel_label(ax, l)
    axes[0, 1].set_title("log1p + median/IQR lens", fontsize=11)
    ax = axes[1, 0]
    for k, (lens_name, ls, mk) in enumerate((("raw_z", "--", "o"), ("robust", "-", "s"))):
        sub = C[C["lens"] == lens_name].set_index("stage").loc[["D0 clean", "Before (D3)", "After"]]
        ax.plot(range(3), sub["hopkins"], ls=ls, marker=mk, color=vs.NEUTRAL["ink"] if lens_name == "raw_z" else vs.PALETTE["blue"], label={"raw_z": "z-score on raw", "robust": "log1p + median/IQR"}[lens_name])
    ax.set_xticks(range(3)); ax.set_xticklabels(["D0 clean", "Before (D3)", "After"]); ax.set_ylabel("Hopkins statistic"); ax.legend(); vs.panel_label(ax, "c")
    ax = axes[1, 1]
    for lens_name, ls, mk in (("raw_z", "--", "o"), ("robust", "-", "s")):
        sub = C[C["lens"] == lens_name].set_index("stage").loc[["D0 clean", "Before (D3)", "After"]]
        ax.plot(range(3), sub["silhouette_hidden_labels"], ls=ls, marker=mk, color=vs.NEUTRAL["ink"] if lens_name == "raw_z" else vs.PALETTE["blue"], label={"raw_z": "z-score on raw", "robust": "log1p + median/IQR"}[lens_name])
    ax.set_xticks(range(3)); ax.set_xticklabels(["D0 clean", "Before (D3)", "After"]); ax.set_ylabel("Silhouette of hidden segments (diagnostic)"); ax.legend(); vs.panel_label(ax, "d")
    vs.finalize_figure(fig, FIG, "figE3_structure_vs_clean")

    fig, axes = plt.subplots(1, 3, figsize=(23, 9), gridspec_kw={"width_ratios": [1, 1, 0.04]})
    lim = max(float(np.abs((d3[num].astype(float).corr() - P0).to_numpy()).max()), 0.05)
    for ax, nm, l in ((axes[0], "Before (D3)", "a"), (axes[1], "After", "b")):
        M = stages[nm][num].astype(float).corr() - P0
        im = ax.imshow(M.to_numpy(), cmap=vs.CMAP_DIV, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=45, ha="right"); ax.set_yticks(range(len(num))); ax.set_yticklabels(num)
        for i in range(len(num)):
            for j in range(len(num)):
                ax.text(j, i, f"{M.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(M.iloc[i, j]) > .6 * lim else vs.NEUTRAL["ink"])
        ax.set_title(f"Pearson: {nm} minus D0"); vs.panel_label(ax, l)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, cax=axes[2], label="change in Pearson correlation vs D0")
    vs.finalize_figure(fig, FIG, "figE4_pearson_difference_vs_clean", pad=1.5)
    log.info("\nĐã ghi %d hình vào %s. Script chỉ đo; không ghi vào data/.", len(list(FIG.glob('*.png'))), FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
