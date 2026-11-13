"""Prototype (design validation only): sinh thử 4 phân khúc ẩn và kiểm tra khả năng phân biệt."""
import itertools
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

SIZES = {"HighValue": 48, "PromoHunter": 78, "Infrequent": 66, "Browser": 68}
# mean parameters per segment
P = {
    #            orders  aov  recency disc  sess  smin  pages abandon age
    "HighValue":   dict(o=24, aov=85, rec=12,  disc=0.12, sess=14, smin=11, pg=9,  ab=0.25, age=38),
    "PromoHunter": dict(o=14, aov=35, rec=30,  disc=0.78, sess=9,  smin=8,  pg=7,  ab=0.45, age=29),
    "Infrequent":  dict(o=3,  aov=50, rec=150, disc=0.25, sess=2,  smin=5,  pg=4,  ab=0.35, age=42),
    "Browser":     dict(o=2,  aov=40, rec=60,  disc=0.15, sess=18, smin=14, pg=16, ab=0.80, age=26),
}
CAT = {
    "tier":   {"HighValue": [.05, .35, .60], "PromoHunter": [.55, .35, .10], "Infrequent": [.80, .17, .03], "Browser": [.75, .20, .05]},
    "device": {"HighValue": [.50, .40, .10], "PromoHunter": [.75, .15, .10], "Infrequent": [.35, .55, .10], "Browser": [.80, .10, .10]},
    "app":    {"HighValue": .85, "PromoHunter": .70, "Infrequent": .30, "Browser": .75},
    "cat":    {"HighValue": [.30, .10, .30, .10, .20], "PromoHunter": [.10, .45, .05, .35, .05],
               "Infrequent": [.10, .10, .30, .10, .40], "Browser": [.40, .35, .10, .10, .05]},
}


def beta(rng, mean, conc, n):
    return rng.beta(mean * conc, (1 - mean) * conc, n)


def gen(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for seg, n in SIZES.items():
        p = P[seg]
        r = 3.0  # NegBin dispersion
        orders = np.maximum(1, rng.negative_binomial(r, r / (r + p["o"]), n))
        aov = rng.lognormal(np.log(p["aov"]), 0.30, n)
        total = orders * aov * rng.lognormal(0, 0.08, n)
        rec = np.minimum(365, rng.gamma(1.5, p["rec"] / 1.5, n))
        disc = beta(rng, p["disc"], 8, n)
        sess = rng.gamma(4, p["sess"] / 4, n)
        smin = rng.lognormal(np.log(p["smin"]), 0.35, n)
        pg = rng.gamma(5, p["pg"] / 5, n)
        ab = beta(rng, p["ab"], 10, n)
        age = np.clip(rng.normal(p["age"], 8, n), 18, 70)
        d = pd.DataFrame(dict(age=age, n_orders_12m=orders, avg_order_value=aov, total_spend_12m=total,
                              recency_days=rec, discount_usage_rate=disc, sessions_per_month=sess,
                              avg_session_minutes=smin, pages_per_session=pg, cart_abandon_rate=ab))
        d["membership_tier"] = rng.choice(3, n, p=CAT["tier"][seg])
        d["device_type"] = rng.choice(["mobile", "desktop", "tablet"], n, p=CAT["device"][seg])
        d["app_installed"] = (rng.random(n) < CAT["app"][seg]).astype(int)
        d["preferred_category"] = rng.choice(["fashion", "electronics", "home", "beauty", "grocery"], n, p=CAT["cat"][seg])
        d["seg"] = seg
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


SKEWED = ["n_orders_12m", "avg_order_value", "total_spend_12m", "recency_days", "sessions_per_month",
          "avg_session_minutes", "pages_per_session"]
NUM = ["age", "n_orders_12m", "avg_order_value", "total_spend_12m", "recency_days", "discount_usage_rate",
       "sessions_per_month", "avg_session_minutes", "pages_per_session", "cart_abandon_rate"]


def design(df, with_cat=True, drop_total=False):
    x = df[NUM].copy()
    x[SKEWED] = np.log1p(x[SKEWED])
    if drop_total:
        x = x.drop(columns="total_spend_12m")
    if with_cat:
        x = pd.concat([x, df[["membership_tier", "app_installed"]],
                       pd.get_dummies(df[["device_type", "preferred_category"]]).astype(int)], axis=1)
    return StandardScaler().fit_transform(x), x


if __name__ == "__main__":
    df = gen(42)
    y = df["seg"].values
    _, xl = design(df, with_cat=False)

    print("== 1. Eta^2 (ty le phuong sai giai thich boi phan khuc), dac trung so da log1p ==")
    eta = {}
    for c in xl.columns:
        g = xl[c].groupby(y)
        eta[c] = ((g.mean() - xl[c].mean()) ** 2 * g.size()).sum() / ((xl[c] - xl[c].mean()) ** 2).sum()
    print(pd.Series(eta).sort_values(ascending=False).round(2).to_string())

    print("\n== 2. Tung cap phan khuc: LDA 5-fold CV accuracy + 3 dac trung phan biet nhat (|Cohen d|) ==")
    for a, b in itertools.combinations(SIZES, 2):
        m = np.isin(y, [a, b])
        xs = xl[m]; ys = y[m]
        ds = {}
        for c in xs.columns:
            va, vb = xs[c][ys == a], xs[c][ys == b]
            sp = np.sqrt((va.var() + vb.var()) / 2)
            ds[c] = abs(va.mean() - vb.mean()) / sp
        top = ", ".join(f"{k}({v:.1f})" for k, v in sorted(ds.items(), key=lambda t: -t[1])[:3])
        acc = cross_val_score(LinearDiscriminantAnalysis(), StandardScaler().fit_transform(xs), ys, cv=5).mean()
        print(f"{a:>11} vs {b:<11} acc={acc:.2f}  top: {top}")

    print("\n== 3. K-Means (k=4, n_init=10) tren du lieu SACH, 10 seed sinh du lieu ==")
    for name, kw in [("so + cat", dict()), ("chi so", dict(with_cat=False)),
                     ("so + cat, bo total_spend", dict(drop_total=True))]:
        aris, nmis, sils = [], [], []
        for s in range(10):
            d = gen(100 + s)
            X, _ = design(d, **kw)
            lab = KMeans(4, n_init=10, random_state=0).fit_predict(X)
            aris.append(adjusted_rand_score(d["seg"], lab)); nmis.append(normalized_mutual_info_score(d["seg"], lab))
            sils.append(silhouette_score(X, lab))
        print(f"{name:<26} ARI={np.mean(aris):.2f}±{np.std(aris):.2f}  NMI={np.mean(nmis):.2f}  Sil={np.mean(sils):.2f}")

    print("\n== 4. Chon k bang silhouette tren seed 42 (so + cat) ==")
    X, _ = design(df)
    for k in range(2, 8):
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
        print(k, round(silhouette_score(X, lab), 3))

    print("\n== 5. Ma tran nham lan K-Means k=4 (seed 42) ==")
    lab = KMeans(4, n_init=10, random_state=0).fit_predict(X)
    print(pd.crosstab(df["seg"], lab).to_string())

    print("\n== 6. Tuong quan |r| > 0.7 (da log1p) ==")
    cm = xl.corr().abs()
    for a, b in itertools.combinations(cm.columns, 2):
        if cm.loc[a, b] > 0.7:
            print(f"{a} ~ {b}: {xl.corr().loc[a, b]:.2f}")
