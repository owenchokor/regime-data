"""7차(진단): 신호-체결 시점 괴리.
 (a) 체결 지연 d: 월말 t 신호를 t+d 거래일 종가에 체결, 다음 월말+d 까지 보유 (d=0 이 기준형)
 (b) 선행 분석 s: 월말 s 거래일 전 점수로 20개를 미리 뽑고 월말 종가 체결 → 미리 분석해도 되는지
     (유니버스는 t월말 정의 사용 — 월말 몇 일치 정보 선반영, 영향 미미하다고 가정·한계)"""
import numpy as np, pandas as pd
from variants import Sim, COST
from exp2 import rmetrics, rf, r, gates
import stocks
sim = Sim(); S = sim.S; gate = gates()["PxMA10"]; ra = sim.ra
c = stocks.load_panels()[0]
idx = (1 + ra).cumprod().where(c.notna()); score = idx / idx.rolling(252, min_periods=200).max()
U = S["elig"] & (S["rank"] <= 500)
days = ra.index; per = days.to_period("M")
me = pd.Series(days).groupby(per).max(); pos = {d: i for i, d in enumerate(days)}
def top20(t, s):
    d = days[pos[me[t]] - s]
    sc = score.loc[d].where(U.loc[t].reindex(score.columns).fillna(False))
    return set(sc.nlargest(20).index)
def run(d_exec=0, s_lead=0):
    out = {}; prev = set()
    for t in U.index:
        if t + 1 not in me.index or pd.isna(gate.get(t)) or t < pd.Period("2015-12", "M"): continue
        i0, i1 = pos[me[t]] + d_exec, pos[me[t + 1]] + d_exec
        if i1 >= len(days): continue
        g = gate[t]; rfm = float(rf.get(t + 1, 0.0))
        if g == 0: out[t + 1] = rfm; prev = set(); continue
        tk = list(top20(t, s_lead))
        R = ra.iloc[i0 + 1:i1 + 1][tk].values
        gross = (np.prod(1 + R, axis=0) - 1).mean()
        turn = 2 * len(set(tk) - prev) / 20 if prev else 1.0
        out[t + 1] = gross - turn * COST / 1e4; prev = set(tk)
    return pd.Series(out).reindex(r.index)
res = {("d", d): run(d_exec=d) for d in [0, 1, 3, 5, 10]}
res.update({("s", s): run(s_lead=s) for s in [3, 5, 10]})
jac = {s: np.mean([len(top20(t, 0) & top20(t, s)) / 20 for t in U.index[24:-1]]) for s in [3, 5, 10]}
rows = []
for k, x in res.items():
    a = rmetrics(x.loc["2019-01":"2024-06"], rf); b = rmetrics(x.loc["2016-01":"2026-08"], rf)
    rows.append([f"{k[0]}={k[1]}", a["CAGR"], a["Sharpe"], a["MDD"], b["CAGR"], b["Sharpe"], b["MDD"]])
T = pd.DataFrame(rows, columns=["변형", "WF_CAGR", "WF_Sh", "WF_MDD", "ALL_CAGR", "ALL_Sh", "ALL_MDD"]).set_index("변형")
pd.set_option("display.width", 200); print(T.round(3)); print("월말 top20 과 s일 전 top20 평균 중복률", {k: round(v, 2) for k, v in jac.items()})
pd.to_pickle(dict(res=res, jac=jac), "run7.pkl")
