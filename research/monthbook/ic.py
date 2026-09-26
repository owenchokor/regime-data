import pickle, numpy as np, pandas as pd
X = pickle.load(open("feats_ex.pkl", "rb")); F = X["F"]
FEATS = [c for c in F.columns if c not in ("sector", "ret", "name", "t", "ticker", "zero_vol_days")]
def ic_table(F, feats):
    F = F.sort_values(["t", "ticker"])   # 동점 순서 누설 차단
    out = {}
    for f in feats:
        ics = F.groupby("t").apply(lambda g: g[f].rank().corr(g["ret"].rank()) if g[f].notna().sum() > 5 else np.nan).dropna()
        yr = pd.Series([p.year for p in ics.index], index=ics.index)
        a, b = ics[yr <= 2020], ics[yr >= 2021]
        # 상위5 - 하위5 (피처 기준) 평균 수익차
        spr = F.groupby("t").apply(lambda g: g.nlargest(5, f)["ret"].mean() - g.nsmallest(5, f)["ret"].mean() if g[f].notna().sum() > 10 else np.nan).dropna()
        out[f] = dict(IC=ics.mean(), t=ics.mean() / ics.std() * np.sqrt(len(ics)), hit=(ics > 0).mean(), IC_16_20=a.mean(), IC_21_26=b.mean(),
                      top5_bot5=spr.mean(), n=len(ics))
    return pd.DataFrame(out).T.sort_values("t")
if __name__ == "__main__":
    T = ic_table(F, FEATS); pd.set_option("display.width", 200); print(T.round(3).to_string())
