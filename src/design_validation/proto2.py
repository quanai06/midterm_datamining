"""Vong 2: them chong lan (khach hang 'cau noi', phan tan lon hon, tuong quan trong cum)."""
import itertools, sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from proto_separability import SIZES, P, CAT, NUM, SKEWED, design

SEGS = list(SIZES)
LOGP = ["o", "aov", "rec", "sess", "smin", "pg"]
LINP = ["disc", "ab", "age"]


def gen2(seed, bridge=0.18, sd_scale=1.0, cat_soft=0.35):
    rng = np.random.default_rng(seed)
    seg = np.concatenate([[s] * n for s, n in SIZES.items()])
    N = len(seg)
    # tham so trung binh theo tung khach: tron voi phan khuc 'doi tac' voi xac suat bridge
    partner = np.array([rng.choice([t for t in SEGS if t != s]) for s in seg])
    is_bridge = rng.random(N) < bridge
    w = np.where(is_bridge, rng.uniform(0.35, 0.65, N), 1.0)
    prm = {}
    for k in LOGP:
        a = np.log([P[s][k] for s in seg]); b = np.log([P[t][k] for t in partner])
        prm[k] = np.exp(w * a + (1 - w) * b)
    for k in LINP:
        a = np.array([P[s][k] for s in seg]); b = np.array([P[t][k] for t in partner])
        prm[k] = w * a + (1 - w) * b
    # nhan to 'hoat dong' tiem an trong cum -> tuong quan giua sessions/pages/orders
    z = rng.normal(0, 1, N)
    orders = np.maximum(1, np.round(prm["o"] * np.exp(0.45 * sd_scale * z + rng.normal(0, 0.35 * sd_scale, N))))
    aov = prm["aov"] * np.exp(rng.normal(0, 0.40 * sd_scale, N))
    total = orders * aov * np.exp(rng.normal(0, 0.10, N))
    rec = np.minimum(365, prm["rec"] * rng.gamma(1.5, 1 / 1.5, N) * np.exp(-0.2 * z))
    sess = prm["sess"] * np.exp(0.40 * sd_scale * z + rng.normal(0, 0.35 * sd_scale, N))
    smin = prm["smin"] * np.exp(rng.normal(0, 0.42 * sd_scale, N))
    pg = prm["pg"] * np.exp(0.2 * z + rng.normal(0, 0.38 * sd_scale, N))
    disc = np.clip(prm["disc"] + rng.normal(0, 0.15 * sd_scale, N), 0.01, 0.99)
    ab = np.clip(prm["ab"] + rng.normal(0, 0.13 * sd_scale, N), 0.02, 0.98)
    age = np.clip(prm["age"] + rng.normal(0, 9, N), 18, 70)
    df = pd.DataFrame(dict(age=age, n_orders_12m=orders, avg_order_value=aov, total_spend_12m=total,
                           recency_days=rec, discount_usage_rate=disc, sessions_per_month=sess,
                           avg_session_minutes=smin, pages_per_session=pg, cart_abandon_rate=ab))

    def soft(p):  # lam mem xac suat phan loai bang cach tron voi phan phoi deu
        p = np.array(p); return (1 - cat_soft) * p + cat_soft / len(p)
    tier = np.array([rng.choice(3, p=soft(CAT["tier"][s])) for s in seg])
    dev = np.array([rng.choice(["mobile", "desktop", "tablet"], p=soft(CAT["device"][s])) for s in seg])
    cat = np.array([rng.choice(["fashion", "electronics", "home", "beauty", "grocery"], p=soft(CAT["cat"][s])) for s in seg])
    app = np.array([rng.random() < (1 - cat_soft) * CAT["app"][s] + cat_soft * 0.5 for s in seg]).astype(int)
    df["membership_tier"] = tier; df["device_type"] = dev; df["app_installed"] = app; df["preferred_category"] = cat
    df["seg"] = seg; df["bridge"] = is_bridge
    return df


def report(**kw):
    print(f"--- tham so: {kw}")
    res = {"so + cat": [], "chi so": []}
    for s in range(10):
        d = gen2(100 + s, **kw)
        for name, k2 in [("so + cat", {}), ("chi so", dict(with_cat=False))]:
            X, _ = design(d, **k2)
            lab = KMeans(4, n_init=10, random_state=0).fit_predict(X)
            res[name].append((adjusted_rand_score(d["seg"], lab), normalized_mutual_info_score(d["seg"], lab),
                              silhouette_score(X, lab)))
    for n, v in res.items():
        a = np.array(v)
        print(f"{n:<9} ARI={a[:,0].mean():.2f}±{a[:,0].std():.2f}  NMI={a[:,1].mean():.2f}  Sil={a[:,2].mean():.2f}")


if __name__ == "__main__":
    for kw in [dict(bridge=0.18, sd_scale=1.0), dict(bridge=0.20, sd_scale=1.2), dict(bridge=0.25, sd_scale=1.3)]:
        report(**kw)
