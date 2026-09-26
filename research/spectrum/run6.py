import pickle, time, numpy as np, pandas as pd
from spectrum import *
sim = Sim(); S = sim.S; gate = gates()["PxMA10"]
W52 = sim.select({})
U = S["elig"] & (S["rank"] <= 500)
rng0 = np.random.default_rng(42)
def rand_pool():
    sc = pd.DataFrame(rng0.random(U.shape), index=U.index, columns=U.columns).where(U)
    rr = sc.rank(axis=1, method="first"); w = (rr <= 20).astype(float)
    return w.div(w.sum(1).replace(0, np.nan), axis=0)
ICS = [-1, -0.3, 0, 0.1, 0.2, 0.3, 0.5, 1]; KS = [3, 5, 10, 20]; REPS = 40
out = []; t0 = time.time()
for pool in ["52WH", "RAND"]:
    for k in KS:
        for ic in ICS:
            reps = 1 if (abs(ic) == 1 or k == 20) else REPS
            for rep in range(reps):
                rng = np.random.default_rng(1000 * rep + k)
                P = W52 if pool == "52WH" else rand_pool()
                x = run(P, S["fwd"], gate, ic, k, rng)
                for per, (a, b) in {"WF": ("2019-01", "2024-06"), "ALL": ("2016-01", "2026-08")}.items():
                    m = rmetrics(x.loc[a:b], rf)
                    kx = r.loc[a:b]; kc = (1 + kx).prod() ** (12 / len(kx)) - 1
                    out.append(dict(pool=pool, k=k, ic=ic, rep=rep, per=per, CAGR=m["CAGR"], Sharpe=m["Sharpe"], MDD=m["MDD"], exKOSPI=m["CAGR"] - kc))
    print(pool, round(time.time() - t0), flush=True)
D = pd.DataFrame(out); D.to_pickle("run6.pkl")
