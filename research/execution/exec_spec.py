"""8차(진단): 체결 현실화 재선별 스펙트럼.
 A: 월말 저녁 20개 채점 → 월말+1거래일 종가에 k개만 매수, 다음 월말+1 까지 보유
 B: 월말+1 종가에 20개 전부 매수 → 평가 완료(월말+1+L) 종가에 탈락분 매도, 매도대금을 선택 k개에 재배분
 IC: 판단 점수와 '보유 구간 실현수익' 순위의 상관(A: [d1,다음 d1], B: [d1+L, 다음 d1]).
 기준 비교: 당일 체결 20개(백테스트 가정), 다음날 체결 20개. 비용 편도 50bp × |Δw|. 진단용."""
import numpy as np, pandas as pd
from scipy.stats import norm
from variants import Sim, COST
from exp2 import rmetrics, rf, r, gates
sim = Sim(); S = sim.S; gate = gates()["PxMA10"]; ra = sim.ra
W = sim.select({})
days = ra.index; per = days.to_period("M")
me = pd.Series(days).groupby(per).max(); pos = {d: i for i, d in enumerate(days)}
RA = ra.values; col = {c: j for j, c in enumerate(ra.columns)}
def seg(i0, i1, js):          # (i0, i1] 누적 총수익
    return np.prod(1 + RA[i0 + 1:i1 + 1][:, js], axis=0) - 1
def pick(f, ic, k, rng):
    z = norm.ppf((pd.Series(f).rank(method="first").values - 0.5) / len(f))
    s = ic * z + np.sqrt(max(0, 1 - ic ** 2)) * rng.standard_normal(len(f))
    return np.argsort(-s)[:k]
def run(mode, ic=0.0, k=20, L=2, d=1, seed=0):
    rng = np.random.default_rng(seed); out = {}; prev = {}
    for t in W.index:
        if t + 1 not in me.index or pd.isna(gate.get(t)) or t < pd.Period("2015-12", "M"): continue
        i0 = pos[me[t]] + d; i1 = pos[me[t + 1]] + d
        if i1 >= len(days): continue
        if gate[t] == 0:
            out[t + 1] = float(rf.get(t + 1, 0.0)); prev = {}; continue
        w = W.loc[t].dropna(); tk = list(w[w > 0].index); js = [col[x] for x in tk]
        if mode == "A":
            f = seg(i0, i1, js); p = pick(f, ic, k, rng) if k < 20 else np.arange(20)
            new = {tk[j]: 1 / len(p) for j in p}
            tr = sum(abs(new.get(x, 0) - prev.get(x, 0)) for x in set(new) | set(prev))
            out[t + 1] = f[p].mean() - tr * COST / 1e4; prev = new
        else:                                   # B
            iL = min(i0 + L, i1)
            f1 = seg(i0, iL, js); f2 = seg(iL, i1, js)
            p = pick(f2, ic, k, rng)
            full = {x: 1 / 20 for x in tk}
            tr1 = sum(abs(full.get(x, 0) - prev.get(x, 0)) for x in set(full) | set(prev))
            v = 1 + f1                            # 종목별 L일 후 가치
            V = v.mean()
            wL = v / v.sum()                      # 드리프트 비중
            tgt = np.zeros(20); tgt[p] = 1 / len(p)
            tr2 = np.abs(tgt - wL).sum()
            out[t + 1] = V * (1 + f2[p].mean()) - 1 - (tr1 + tr2 * V) * COST / 1e4
            prev = {tk[j]: 1 / len(p) for j in p}
    return pd.Series(out).reindex(r.index)
