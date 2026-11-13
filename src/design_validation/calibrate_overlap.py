import importlib.util, copy, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
spec = importlib.util.spec_from_file_location("gen", "src/01_generate_data.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
BASE_P = copy.deepcopy(g.P); BASE_M = g._TRAIT_MEAN_BASE.copy(); SD = g.TRAIT_SD.copy()

def run(label, scale=(1,1.3,1.3,1.3), P_over=None, base_over=None, sd=None):
    g.P.clear(); g.P.update(copy.deepcopy(BASE_P)); g.P.update(P_over or {})
    m = BASE_M.copy()
    for (r, c), v in (base_over or {}).items(): m[r, c] = v
    g.TRAIT_MEAN = m.mean(0) + np.array(scale) * (m - m.mean(0)); g.TRAIT_SD = SD if sd is None else sd
    aris, accs, meds = [], [], []
    for s in range(100, 108):
        d0, t = g.generate_clean(np.random.default_rng(np.random.SeedSequence(s).spawn(4)[0]))
        aris.append(adjusted_rand_score(t.true_segment, KMeans(4, n_init=10, random_state=0).fit_predict(g._design_matrix(d0))))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        accs.append({c: cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), d0[[c]].astype(float), t.true_segment, cv=cv).mean() for c in g.NUMERIC})
        meds.append(d0.assign(seg=t.true_segment.values).groupby("seg")[["n_orders_12m", "total_spend_12m", "sessions_per_month", "pages_per_session", "recency_days"]].median().loc[g.SEGMENTS])
    a = pd.DataFrame(accs).mean(); med = sum(meds) / len(meds)
    print(f"\n### {label}\nARI={np.mean(aris):.2f}±{np.std(aris):.2f}  max acc 1-feat: {a.idxmax()}={a.max():.2f} (top3: {', '.join(f'{k}={v:.2f}' for k, v in a.sort_values(ascending=False).head(3).items())})")
    print(med.round(1).to_string())

if __name__ == "__main__":
    run("E gap 1.0, conv0 -2.6", scale=(1, 1.3, 1.3, 1.3), P_over=dict(conv0=-2.6))
    run("E gap 0.8, conv0 -2.6", scale=(0.8, 1.3, 1.3, 1.3), P_over=dict(conv0=-2.6))
    run("E gap 0.8, conv0 -2.6, Browser I -2.0", scale=(0.8, 1.3, 1.3, 1.3), P_over=dict(conv0=-2.6), base_over={(3, 1): -2.0})
