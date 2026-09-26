"""월별 후보 20개 사후 성과북 데이터: 결정월 t 말 신호 → t말+1거래일 종가 매수 → 다음 월말+1거래일 종가 (A안 체결)."""
import pickle, numpy as np, pandas as pd
from variants import Sim
from exp2 import gates
sim = Sim(); W = sim.select({}); gate = gates()["PxMA10"]; ra = sim.ra
days = ra.index; per = days.to_period("M")
me = pd.Series(days).groupby(per).max(); pos = {d: i for i, d in enumerate(days)}
sec = pd.read_parquet("/home/claude/data/sector_monthly.parquet"); sec["p"] = pd.to_datetime(sec["date"]).dt.to_period("M")
desc = pd.read_csv("/home/claude/data/desc.csv", dtype=str).set_index("Code")
dl = pd.read_csv("/home/claude/data/delist.csv", dtype=str).drop_duplicates("Symbol", keep="last").set_index("Symbol")
k = pd.read_parquet("/home/claude/data/kospi.parquet")["종가"]; k.index = pd.to_datetime(k.index)
book = {}
for t in W.index:
    if t < pd.Period("2016-01", "M") or t + 1 not in me.index or gate.get(t) != 1: continue
    i0, i1 = pos[me[t]] + 1, pos[me[t + 1]] + 1
    if i1 >= len(days): continue
    w = W.loc[t].dropna(); tk = list(w[w > 0].index)
    D = days[i0 + 1:i1 + 1]
    path = (1 + ra.loc[D, tk]).cumprod() - 1
    fin = path.iloc[-1].sort_values(ascending=False)
    dd = ((1 + path) / (1 + path).cummax().clip(lower=1) - 1).min()
    s = sec[sec.p == sec.p[sec.p <= t].max()].set_index("ticker")
    rows = []
    for rank, x in enumerate(fin.index, 1):
        ind = desc.Industry.get(x) if x in desc.index else dl.Industry.get(x)
        prod = desc.Products.get(x) if x in desc.index else None
        rows.append(dict(rank=rank, ticker=x, name=s["종목명"].get(x, desc.Name.get(x, dl.Name.get(x, "") if x in dl.index else "")),
                         sector=s["업종명"].get(x, ""), industry=ind if isinstance(ind, str) else "",
                         products=prod if isinstance(prod, str) else "", ret=fin[x], mdd=dd[x],
                         delisted=x in dl.index and x not in desc.index))
    kp = k.reindex(days[i0:i1 + 1]); kret = (kp / kp.iloc[0] - 1).iloc[1:]
    book[t] = dict(buy=days[i0], sell=days[i1], table=pd.DataFrame(rows), path=path[fin.index], kospi=kret)
pickle.dump(book, open("monthbook.pkl", "wb"))
print(len(book), "개월", min(book), "~", max(book))
t = max(book); print(book[t]["buy"].date(), book[t]["sell"].date()); print(book[t]["table"][["rank","name","sector","products","ret","mdd"]].round(3).to_string())
