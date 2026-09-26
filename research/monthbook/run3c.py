import pickle, pandas as pd
from variants import Sim, turnover_cost, select_liq
from exp2 import rmetrics, rf, r, gates
from core import DEV_END, HOLD_START, LAST_MONTH
gate = gates()["PxMA10"]; sim = Sim()
amount = pd.read_parquet("/home/claude/data/panel_amount.parquet"); amount.index = pd.to_datetime(amount.index)
WF = ("2019-01", DEV_END); HO = (HOLD_START, LAST_MONTH)
def run(W):
    mr, _ = sim.month_returns(W, {}, gate, rf)
    return (mr - turnover_cost(W, gate).reindex(mr.index).fillna(0)).reindex(r.index)
base = run(sim.select({})); res = {c: run(select_liq(sim, c, amount)) for c in (0.30, 0.2, 0.4)}
b = rmetrics(base.loc[WF[0]:WF[1]], rf)
for c, x in res.items():
    m = rmetrics(x.loc[WF[0]:WF[1]], rf); h = rmetrics(x.loc[HO[0]:HO[1]], rf)
    print(c, {k: round(m[k], 3) for k in ["CAGR","Sharpe","MDD","CVaR5"]}, "HO", {k: round(h[k], 3) for k in ["Sharpe","MDD"]})
m = rmetrics(res[0.30].loc[WF[0]:WF[1]], rf)
c1 = m["Sharpe"] >= b["Sharpe"] - 0.05; c2 = (m["MDD"] - b["MDD"] >= 0.03) or (m["CVaR5"] - b["CVaR5"] >= 0.03)
c3 = all(rmetrics(res[c].loc[WF[0]:WF[1]], rf)["Sharpe"] >= b["Sharpe"] - 0.05 for c in (0.2, 0.4))
print("base", round(b["Sharpe"],3), "V8 ①", c1, "②", c2, "③", c3, "→", "통과" if c1 and c2 and c3 else "탈락")
pickle.dump({"main": {"V8 유동성 하위30% 제외": res[0.30]}, "nb": res}, open("run3c.pkl", "wb"))
