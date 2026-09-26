"""KOSPI 월간 타이밍 연구 — 데이터·피처·백테스트 공통 모듈."""
from __future__ import annotations
import numpy as np
import pandas as pd

DATA = "../../data"  # release data-latest의 kospi/kosdaq.parquet

# ── 사전 등록 기간 (결과 보기 전 고정) ──
EVAL_START = "2016-01"      # 24M 피처가 모두 계산되는 첫 수익월
DEV_END = "2024-06"         # 개발 구간 끝
HOLD_START = "2024-07"      # 최종 holdout 시작 (26개월 ≈ 20%)
LAST_MONTH = "2026-08"      # 2026-09는 미완성 월 → 제외
WF_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]  # 2024는 1~6월만
ROLL_WIN = 36               # rolling window 길이(월) — 첫 fold의 expanding 길이와 동일
COSTS_BPS = [0, 10, 20, 50]
SEL_COST_BPS = 20           # nested 선택 시 사용하는 비용


def load_daily() -> pd.DataFrame:
    k = pd.read_parquet(f"{DATA}/kospi.parquet")["종가"].rename("kospi")
    q = pd.read_parquet(f"{DATA}/kosdaq.parquet")["종가"].rename("kosdaq")
    d = pd.concat([k, q], axis=1)
    d.index = pd.to_datetime(d.index)
    return d


def to_monthly(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(d.index.to_period("M"))
    m = g.last()
    m["last_day"] = g.apply(lambda x: x.index.max())
    m["first_day"] = g.apply(lambda x: x.index.min())
    m["n_days"] = g.size()
    m = m.loc[:LAST_MONTH]
    return m


def features(m: pd.DataFrame) -> pd.DataFrame:
    """모든 피처는 t월말까지의 종가만 사용."""
    p, q = m["kospi"], m["kosdaq"]
    r = p.pct_change()
    f = pd.DataFrame(index=m.index)
    for L in [1, 2, 3, 6, 9, 12, 18, 24]:
        f[f"mom{L}"] = p / p.shift(L) - 1
        f[f"qmom{L}"] = q / q.shift(L) - 1
    for N in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 24]:
        f[f"ma{N}"] = p.rolling(N).mean()
        f[f"pma{N}"] = np.log(p / f[f"ma{N}"])
    for N in [6, 12]:
        f[f"qpma{N}"] = np.log(q / q.rolling(N).mean())
        f[f"maslope{N}"] = f[f"ma{N}"] / f[f"ma{N}"].shift(1) - 1
    f["ma3_12"] = np.log(f["ma3"] / f["ma12"])
    f["ma6_12"] = np.log(f["ma6"] / f["ma12"])
    f["ma12_24"] = np.log(f["ma12"] / f["ma24"])
    for W in [3, 6, 12]:
        f[f"vol{W}"] = r.rolling(W).std() * np.sqrt(12)
    f["volchg"] = np.log(f["vol3"] / f["vol12"])
    f["dd12"] = p / p.rolling(12).max() - 1
    f["ddall"] = p / p.cummax() - 1
    f["dlow12"] = p / p.rolling(12).min() - 1
    for L in [1, 3, 6, 12]:
        f[f"rel{L}"] = f[f"mom{L}"] - f[f"qmom{L}"]
    f["relpos12"] = np.log((p / q) / (p / q).rolling(12).mean())
    f["acc3"] = f["mom3"] - f["mom3"].shift(3)
    return f


# ML용 소수 피처 (사전 지정; 폴드 내부에서 상관 필터 + 정규화로 추가 축소)
ML_FEATS = ["mom1", "mom3", "mom6", "mom12", "pma12", "ma3_12",
            "vol6", "volchg", "dd12", "rel6", "acc3"]


# ── 백테스트 ──
def strat_returns(pos: pd.Series, r: pd.Series, cost_bps: float) -> pd.Series:
    """pos[t]: t월말 결정 → r[t+1]에 적용. 비용은 포지션 변경분에만."""
    p = pos.reindex(r.index)
    held = p.shift(1)
    turn = held.diff().abs()
    # 첫 유효 포지션 진입은 현금에서의 매수로 간주
    first = held.notna() & held.shift(1).isna()
    turn[first] = held[first].abs()
    return held * r - turn * cost_bps / 1e4


def metrics(x: pd.Series, pos: pd.Series | None = None) -> dict:
    x = x.dropna()
    n = len(x)
    eq = (1 + x).cumprod()
    cagr = eq.iloc[-1] ** (12 / n) - 1
    vol = x.std(ddof=1) * np.sqrt(12)
    mdd = (eq / eq.cummax().clip(lower=1) - 1).min()
    dd_dev = np.sqrt((np.minimum(x, 0) ** 2).mean()) * np.sqrt(12)
    out = dict(n=n, CAGR=cagr, Cum=eq.iloc[-1] - 1, Vol=vol,
               Sharpe=x.mean() / x.std(ddof=1) * np.sqrt(12) if vol > 0 else np.nan,
               MDD=mdd, Calmar=cagr / abs(mdd) if mdd < 0 else np.nan,
               WinRate=(x > 0).mean(), MeanM=x.mean(), MedianM=x.median(),
               DownDev=dd_dev, Worst=x.min(), Best=x.max())
    if pos is not None:
        p = pos.shift(1).reindex(x.index)
        out["Exposure"] = p.mean()
        out["TurnoverYr"] = p.diff().abs().sum() / n * 12
    return out


def sharpe(x: np.ndarray) -> float:
    s = np.std(x, ddof=1)
    return np.mean(x) / s * np.sqrt(12) if s > 0 else np.nan
