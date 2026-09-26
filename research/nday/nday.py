"""5차: n거래일 리밸런싱 변형 (사전등록 — 이 파일 커밋 후 실행).

변형 (임의값):  N5  n=5 거래일   이웃 4/6
               N10 n=10 거래일  이웃 8/12
정의: 리밸런싱일 t 종가에 52WH(수정가/252일 고가) 상위 20 동일가중으로 전량 재조정.
  유니버스·게이트는 t 이전 마지막 월말 값(기준형과 동일 정의, 룩어헤드 없음) → 리밸런싱 빈도 효과만 분리.
  비용: 편도 50bp × |목표비중 − 드리프트비중| 합. 현금 = CD 월평균 일할.
  위상: 평가 시작 첫 거래일부터 n일 간격 (위상 민감도는 한계로 명시).
검증: 동일 엔진에 '월말 리밸런싱'을 넣어 기준형(Sim) 수치 재현 여부 확인.
판정: 일별 수익을 월로 합성 → 3차와 동일 기준 (WF-OOS 2019-01~2024-06, 초과수익 Sharpe)
  ① Sharpe ≥ 기준형 − 0.05  ② MDD 또는 CVaR5 3%p 이상 개선  ③ 이웃값에서 ① 유지
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TOPN, UNIV, COST = 20, 500, 50
ND_VARIANTS = {"N5": 5, "N10": 10}
ND_NEIGHBORS = {"N5": [4, 6], "N10": [8, 12]}


def daily_engine(ra: pd.DataFrame, c: pd.DataFrame, S: dict, gate_m: pd.Series, rf_m: pd.Series,
                 rebal: pd.DatetimeIndex, start="2015-12-01") -> tuple[pd.Series, pd.Series]:
    """반환: (일별 순수익, 일별 회전율)."""
    idx = (1 + ra).cumprod().where(c.notna())
    score = idx / idx.rolling(252, min_periods=200).max()
    per = ra.index.to_period("M")
    U = S["elig"] & (S["rank"] <= UNIV)
    days = ra.index[ra.index >= start]
    last_me = {}   # 각 거래일 → 그 날 이전(포함) 마지막 '완료' 월말의 월
    me = ra.groupby(per).apply(lambda x: x.index.max())
    me_set = set(me.values)
    for d in days:
        p = d.to_period("M")
        last_me[d] = p if d in me_set else p - 1
    rb = set(rebal)
    R = ra.loc[days].values; cols = ra.columns
    w = np.zeros(len(cols)); out = np.zeros(len(days)); turn = np.zeros(len(days))
    ndays = pd.Series(per).groupby(per).size()
    for i, d in enumerate(days):
        p = d.to_period("M")
        cash = 1 - w.sum()
        rfd = float(rf_m.get(p, 0.0)) / ndays[p]
        gross = w @ R[i] + cash * rfd if i > 0 else 0.0
        if i > 0:
            w = w * (1 + R[i]) / (1 + gross)
        if d in rb:
            q = last_me[d]
            g = gate_m.get(q, np.nan)
            tgt = np.zeros(len(cols))
            if g == 1 and q in U.index:
                sc = score.loc[d].where(U.loc[q].reindex(cols).fillna(False).values & c.loc[d].notna().values)
                top = sc.nlargest(TOPN).index
                tgt[cols.get_indexer(top)] = 1.0 / len(top) if len(top) else 0
            tr = np.abs(tgt - w).sum()
            turn[i] = tr; w = tgt
            gross -= tr * COST / 1e4
        out[i] = gross
    return pd.Series(out, index=days), pd.Series(turn, index=days)


def rebal_days(days: pd.DatetimeIndex, n: int | str, start="2015-12-01") -> pd.DatetimeIndex:
    d = days[days >= start]
    if n == "M":
        return pd.DatetimeIndex(pd.Series(d).groupby(d.to_period("M")).max().values)
    return d[::n]
