import time, numpy as np, pandas as pd
from exec_spec import *
rows = []; t0 = time.time()
def rec(name, ic, k, x, rep):
    for pr, (a, b) in {"WF": ("2019-01", "2024-06"), "ALL": ("2016-01", "2026-08")}.items():
        m = rmetrics(x.loc[a:b], rf); kx = r.loc[a:b]; kc = (1 + kx).prod() ** (12 / len(kx)) - 1
        rows.append(dict(mode=name, ic=ic, k=k, rep=rep, per=pr, CAGR=m["CAGR"], Sharpe=m["Sharpe"], MDD=m["MDD"], ex=m["CAGR"] - kc))
rec("당일20", None, 20, run("A", d=0), 0); rec("익일20", None, 20, run("A", d=1), 0)
for k in [5, 10]:
    for ic in [0, 0.1, 0.2, 0.3, 0.5]:
        for rep in range(30):
            rec("A", ic, k, run("A", ic, k, seed=rep), rep)
            rec("B", ic, k, run("B", ic, k, L=2, seed=rep), rep)
    print(k, round(time.time() - t0), flush=True)
pd.DataFrame(rows).to_pickle("run8.pkl")
