import numpy as np, pandas as pd
exec(open("delay.py").read().split("def run(")[0])
def topn(t, s, n):
    d = days[pos[me[t]] - s]
    return set(score.loc[d].where(U.loc[t].reindex(score.columns).fillna(False)).nlargest(n).index)
T = U.index[24:-1]
for s in [5, 10, 15]:
    for n in [40, 60, 100]:
        cov = np.mean([len(topn(t, 0, 20) & topn(t, s, n)) / 20 for t in T])
        print(f"월말 top20 중 {s}거래일 전 top{n} 에 있던 비율 {cov:.2f}")
# 월 내 누적 워치리스트: 해당 월 어느 날이든 top40 에 한 번이라도 든 종목이 월말 top20 을 덮는 비율
cov = []
for t in T:
    D = days[per == t]; seen = set()
    for d in D[:-1]:
        seen |= set(score.loc[d].where(U.loc[t].reindex(score.columns).fillna(False)).nlargest(40).index)
    cov.append((len(topn(t, 0, 20) & seen) / 20, len(seen)))
cov = np.array(cov); print("월중 누적 top40 워치리스트: 월말 top20 포함률", cov[:, 0].mean().round(2), "평균 워치리스트 크기", cov[:, 1].mean().round(0))
