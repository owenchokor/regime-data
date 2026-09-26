import pickle, time, numpy as np, pandas as pd
from variants import Sim
from exp2 import rmetrics, rf, r, gates
from core import DEV_END, HOLD_START, LAST_MONTH
from nday import *
kg = gates()["PxMA10"]; sim = Sim()
c, _, _, _ = __import__("stocks").load_panels()
ra = sim.ra; days = ra.index
rfm = rf.copy()
WF = ("2019-01", DEV_END); HO = (HOLD_START, LAST_MONTH)
def run(n):
    d, tv = daily_engine(ra, c, sim.S, kg, rfm, rebal_days(days, n, kg))
    m = (1 + d).groupby(d.index.to_period("M")).prod() - 1
    return m.reindex(r.index), d, tv
t0 = time.time(); res = {}
for n in ["M", 5, 10, 4, 6, 8, 12]:
    res[n] = run(n); print(n, round(time.time() - t0), flush=True)
pickle.dump(res, open("run5.pkl", "wb"))
