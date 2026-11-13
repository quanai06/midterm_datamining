import itertools, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from proto2 import gen2
from proto_separability import design, SIZES
df = gen2(42, bridge=0.18, sd_scale=1.0); y = df["seg"].values
X, xl = design(df, with_cat=False)
print("cau noi:", int(df.bridge.sum()), "/", len(df))
print("== eta^2 =="); 
eta = {c: (((xl[c].groupby(y).mean()-xl[c].mean())**2*xl[c].groupby(y).size()).sum()/((xl[c]-xl[c].mean())**2).sum()) for c in xl}
print(pd.Series(eta).sort_values(ascending=False).round(2).to_string())
print("== cap phan khuc ==")
for a,b in itertools.combinations(SIZES,2):
    m=np.isin(y,[a,b]); xs=xl[m]; ys=y[m]
    ds={c:abs(xs[c][ys==a].mean()-xs[c][ys==b].mean())/np.sqrt((xs[c][ys==a].var()+xs[c][ys==b].var())/2) for c in xs}
    top=", ".join(f"{k}({v:.1f})" for k,v in sorted(ds.items(),key=lambda t:-t[1])[:3])
    acc=cross_val_score(LinearDiscriminantAnalysis(),StandardScaler().fit_transform(xs),ys,cv=5).mean()
    print(f"{a:>11} vs {b:<11} acc={acc:.2f} top: {top}")
Xa,_=design(df)
print("== silhouette theo k ==")
for k in range(2,8): print(k, round(silhouette_score(Xa, KMeans(k,n_init=10,random_state=0).fit_predict(Xa)),3))
lab=KMeans(4,n_init=10,random_state=0).fit_predict(Xa)
print(pd.crosstab(df.seg,lab).to_string())
print("ARI cau noi vs khong:", {b: round(adjusted_rand_score(df.seg[df.bridge==b], lab[df.bridge==b]),2) for b in (False,True)})
cm=xl.corr()
print("== |r|>0.6 ==")
for a,b in itertools.combinations(cm.columns,2):
    if abs(cm.loc[a,b])>0.6: print(a,"~",b,round(cm.loc[a,b],2))
print(df.groupby("seg")[["n_orders_12m","avg_order_value","total_spend_12m","recency_days","discount_usage_rate","sessions_per_month","avg_session_minutes","pages_per_session","cart_abandon_rate","age"]].median().round(1).to_string())
print(df[["n_orders_12m","avg_order_value","total_spend_12m","recency_days","sessions_per_month","avg_session_minutes","pages_per_session"]].skew().round(2).to_string())
