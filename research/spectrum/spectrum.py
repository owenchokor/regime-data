"""6차(진단): 재선별 스펙트럼. 후보 20개 중 k개를 '실현수익 순위와 상관 IC'인 판단으로 고를 때의 성과 분포.
IC=1 오라클, 0 무작위, −1 최악. 대조군: 유니버스 무작위 20개 후보풀. 진단용이라 판정 기준 없음."""
import numpy as np, pandas as pd
from scipy.stats import norm
from variants import Sim
from exp2 import rmetrics, rf, r, gates
COST = 50
def run(pool_W, fwd, gate, ic, k, rng):
    rets = {}
    prev = None
    for t in pool_W.index:
        g = gate.get(t, np.nan)
        if pd.isna(g): continue
        p1 = t + 1
        if g == 0:
            rets[p1] = rf.get(p1, 0.0); prev = None; continue
        c = pool_W.loc[t]; c = c[c > 0].index
        if len(c) == 0: continue
        f = fwd.loc[t, c].fillna(0.0).values
        z = norm.ppf((pd.Series(f).rank(method="first").values - 0.5) / len(f))
        s = ic * z + np.sqrt(max(0.0, 1 - ic ** 2)) * rng.standard_normal(len(f))
        pick = np.argsort(-s)[:k]
        w = pd.Series(1.0 / len(pick), index=c[pick])
        tr = w.sub(prev, fill_value=0).abs().sum() if prev is not None else 1.0
        rets[p1] = f[pick].mean() - tr * COST / 1e4
        prev = w
    return pd.Series(rets).reindex(r.index)
