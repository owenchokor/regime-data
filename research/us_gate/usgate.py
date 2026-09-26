"""4차: 52WH × 미국 게이트 선별. 변형·파라미터·기준은 실행 전 고정(이 파일 커밋 = 사전등록).

시점: 한국 t월 마지막 거래일 − 2일(달력) 이전 날짜의 미국 관측치만 사용.
  왜: 한국 월말 장마감(06:30 UTC) 시점엔 당일 미국장이 없고, FRED 금리는 1영업일 지연 공표.
변형 (임계값 전부 임의값):
  U1 NQ 단독        : 나스닥 월말 > 10개월 평균 (KOSPI 게이트 대체)
  U2 KOSPI∧NQ       : PxMA10 AND 나스닥 > 10개월 평균
  U3 KOSPI∧금리급등X : PxMA10 AND (미 10년물 3개월 변화 ≤ +0.50%p)
이웃값: U1·U2 → MA 8/12, U3 → 0.25/0.75%p
통과 기준: 3차와 동일(WF-OOS 2019-01~2024-06, 50bp, 초과수익 Sharpe)
  ① Sharpe ≥ 기준형 − 0.05  ② MDD 또는 CVaR5 3%p 이상 개선  ③ 이웃값에서 ① 유지
holdout(2024-07~)은 진단 전용.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core import DATA, load_daily

LAG_DAYS = 2
US_VARIANTS = {
    "U1 NQ 단독": ("nq_only", 10),
    "U2 KOSPI∧NQ": ("kospi_and_nq", 10),
    "U3 KOSPI∧금리급등X": ("kospi_and_rate", 0.50),
}
US_NEIGHBORS = {
    "U1 NQ 단독": [("nq_only", 8), ("nq_only", 12)],
    "U2 KOSPI∧NQ": [("kospi_and_nq", 8), ("kospi_and_nq", 12)],
    "U3 KOSPI∧금리급등X": [("kospi_and_rate", 0.25), ("kospi_and_rate", 0.75)],
}


def us_monthly(sid: str) -> pd.Series:
    s = pd.read_parquet(f"{DATA}/fred_{sid}.parquet").iloc[:, 0].dropna()
    kd = load_daily()["kospi"].dropna()
    last = kd.groupby(kd.index.to_period("M")).apply(lambda x: x.index.max())
    cut = last - pd.Timedelta(days=LAG_DAYS)
    pos = s.index.searchsorted(cut.values, side="right") - 1
    return pd.Series(np.where(pos >= 0, s.values[pos], np.nan), index=last.index, name=sid)


def us_gate(kind: str, p, kospi_gate: pd.Series) -> pd.Series:
    kg = kospi_gate
    if kind in ("nq_only", "kospi_and_nq"):
        nq = us_monthly("NASDAQCOM")
        g = (nq > nq.rolling(int(p)).mean()).astype(float).where(nq.rolling(int(p)).mean().notna())
        g = g.reindex(kg.index)
        return g if kind == "nq_only" else (g * kg).where(g.notna() & kg.notna())
    if kind == "kospi_and_rate":
        y = us_monthly("DGS10")
        g = ((y - y.shift(3)) <= p).astype(float).where(y.shift(3).notna()).reindex(kg.index)
        return (g * kg).where(g.notna() & kg.notna())
    raise ValueError(kind)
