import pickle, time, numpy as np, pandas as pd
from variants import Sim, turnover_cost
from exp2 import rmetrics, rf, r, gates
from core import DEV_END, HOLD_START, LAST_MONTH
from usgate import *
kg = gates()["PxMA10"]
sim = Sim(); W = sim.select({})
WF = ("2019-01", DEV_END); HO = (HOLD_START, LAST_MONTH)
def run(g, opt={}):
    mr, _ = sim.month_returns(W, opt, g, rf)
    return (mr - turnover_cost(W, g).reindex(mr.index).fillna(0)).reindex(r.index)
t0 = time.time()
G = {"BASE": kg} | {k: us_gate(*v, kg) for k, v in US_VARIANTS.items()}
res = {k: run(g) for k, g in G.items()}
# forward 후보 ②와의 결합도 진단용으로 계산 (판정 아님)
res_stop = {k: run(g, {"stock_stop": 0.10}) for k, g in G.items()}
nb = {(k, str(p)): run(us_gate(*p, kg)) for k, lst in US_NEIGHBORS.items() for p in lst}
print("sim", round(time.time() - t0))
pickle.dump(dict(G=G, res=res, res_stop=res_stop, nb=nb), open("run4.pkl", "wb"))
b = rmetrics(res["BASE"].loc[WF[0]:WF[1]], rf)
rows = []
for k, x in res.items():
    m = rmetrics(x.loc[WF[0]:WF[1]], rf); h = rmetrics(x.loc[HO[0]:HO[1]], rf)
    g = G[k]
    on_wf = g.loc["2018-12":"2024-05"].mean(); on_ho = g.loc["2024-06":"2026-07"].mean()
    if k == "BASE": c = ["–"] * 3; ok = "기준"
    else:
        c1 = m["Sharpe"] >= b["Sharpe"] - 0.05
        c2 = (m["MDD"] - b["MDD"] >= 0.03) or (m["CVaR5"] - b["CVaR5"] >= 0.03)
        c3 = all(rmetrics(nb[(k, str(p))].loc[WF[0]:WF[1]], rf)["Sharpe"] >= b["Sharpe"] - 0.05 for p in US_NEIGHBORS[k])
        c = ["○" if z else "✕" for z in (c1, c2, c3)]; ok = "통과" if (c1 and c2 and c3) else "탈락"
    rows.append([k, m["CAGR"], m["Sharpe"], m["MDD"], m["CVaR5"], on_wf, *c, ok, h["CAGR"], h["Sharpe"], h["MDD"], on_ho])
T = pd.DataFrame(rows, columns=["변형","WF_CAGR","WF_Sh","WF_MDD","WF_CVaR","WF_투자율","①","②","③","판정","HO_CAGR","HO_Sh","HO_MDD","HO_투자율"]).set_index("변형")
pd.set_option("display.width", 250); print(T.round(3))
print("\n이웃 WF Sharpe:"); print({f"{k}{p}": round(rmetrics(v.loc[WF[0]:WF[1]], rf)["Sharpe"], 3) for (k, p), v in nb.items()})
print("\n+10%스톱 결합(진단):"); print(pd.DataFrame({k: {**{f"WF_{a}": rmetrics(v.loc[WF[0]:WF[1]], rf)[a] for a in ["Sharpe","MDD"]}, **{f"HO_{a}": rmetrics(v.loc[HO[0]:HO[1]], rf)[a] for a in ["CAGR","Sharpe","MDD"]}} for k, v in res_stop.items()}).T.round(3))
print("\n2026-05~08 월수익 / 게이트(결정월):")
print(pd.DataFrame({k: v.loc["2026-04":"2026-08"] for k, v in res.items()}).round(3))
print(pd.DataFrame({k: g.loc["2026-03":"2026-07"] for k, g in G.items()}))
