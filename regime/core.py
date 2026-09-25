"""
시장 국면 판정 — 오닐 3국면 (강세/박스권/약세)

판정 기준:
  강세(2): 지수 > MA(N) AND 50일선 위 종목 비율 > b_hi
            AND 분배일 <= dmax AND 환율 상승폭 <= fxmax AND 금리 상승폭 <= rmax
            AND 위 조건이 confirm일 연속 유지
  약세(0): 지수 < MA(N) AND 50일선 위 종목 비율 < b_lo
  박스권(1): 그 외

파라미터는 전부 임의값 — IS 구간 그리드 탐색으로 결정.
정답(y): 향후 FWD 영업일 동안 유니버스 개별주 수익률 중앙값이 상위 1/3이면 1
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

IndexMode = Literal["kospi", "kosdaq", "either", "both"]

@dataclass
class RegimeCFG:
    ma_n:     int       = 120       # 이동평균 기간 (임의값)
    b_hi:     float     = 0.55      # 강세: 50일선 위 종목 비율 하한 (임의값)
    b_lo:     float     = 0.40      # 약세: 50일선 위 종목 비율 상한 (임의값)
    dmax:     int       = 5         # 강세: 최대 분배일 수 (임의값)
    fxmax:    float     = np.inf    # 강세: 원/달러 20일 변화율 상한 (임의값)
    rmax:     float     = np.inf    # 강세: 미국 10년물 20일 변화폭 상한 (임의값)
    idx_mode: IndexMode = "both"    # 지수 기준 (임의값)
    confirm:  int       = 3         # 연속 유지 일수 (임의값)
    fwd:      int       = 20        # 정답 라벨 전방기간 영업일 (임의값)
    fx_lag:   int       = 5         # DEXKOUS 공표 지연 (임의값, 확인 필요)
    rate_lag: int       = 1         # DGS10 공표 지연


def _comb(a: np.ndarray, b: np.ndarray, mode: IndexMode) -> np.ndarray:
    if mode == "kospi":  return a
    if mode == "kosdaq": return b
    if mode == "either": return a | b
    return a & b  # both


def _persist(x: np.ndarray, c: int) -> np.ndarray:
    """c일 연속 True인 마지막 날만 True로 만든다 (잦은 진입 방지)"""
    if c == 1: return x
    return np.convolve(x.astype(int), np.ones(c, int), "full")[:len(x)] == c


def build_regime(
    close_panel: pd.DataFrame,   # 날짜×종목 수정종가
    kospi:       pd.DataFrame,   # get_index_ohlcv_by_date 결과 (종가·거래량 컬럼)
    kosdaq:      pd.DataFrame,
    krw:         pd.Series,      # DEXKOUS 일별 (공표일 기준)
    us10:        pd.Series,      # DGS10 일별
    cfg:         RegimeCFG = RegimeCFG(),
) -> pd.DataFrame:
    """
    Returns
    -------
    DataFrame, index=날짜, columns=[state, p50, fx20, r20, dd_k, dd_q]
      state: 0=약세 1=박스권 2=강세
    """
    d = close_panel.index

    # ── 지수 이동평균 ─────────────────────────────────────────────
    def idx_series(df: pd.DataFrame) -> pd.Series:
        # KRX 지수 컬럼명 확인 필요 — 종가 컬럼이 다를 수 있음
        col = [c for c in df.columns if "종가" in c or c == "Close"]
        if not col:
            raise ValueError(f"종가 컬럼 없음: {df.columns.tolist()}")
        return df[col[0]].reindex(d)

    def vol_series(df: pd.DataFrame) -> pd.Series:
        col = [c for c in df.columns if "거래량" in c or c == "Volume"]
        if not col:
            raise ValueError(f"거래량 컬럼 없음: {df.columns.tolist()}")
        return df[col[0]].reindex(d)

    ki, qi = idx_series(kospi), idx_series(kosdaq)
    kv, qv = vol_series(kospi), vol_series(kosdaq)

    ki_ma = ki.rolling(cfg.ma_n, min_periods=int(cfg.ma_n * 0.8)).mean()
    qi_ma = qi.rolling(cfg.ma_n, min_periods=int(cfg.ma_n * 0.8)).mean()

    up_k = (ki > ki_ma).values
    up_q = (qi > qi_ma).values
    dn_k = (ki < ki_ma).values
    dn_q = (qi < qi_ma).values

    # ── 분배일: 지수 -0.2% 이상 하락 + 거래량 증가, 최근 25일 합산 ──
    def dist_days(c: pd.Series, v: pd.Series) -> np.ndarray:
        ret = c.pct_change()
        flag = ((ret <= -0.002) & (v > v.shift(1))).astype(int)
        return flag.rolling(25, min_periods=1).sum().values

    dd_k = dist_days(ki, kv)
    dd_q = dist_days(qi, qv)
    dd = {"kospi": dd_k, "kosdaq": dd_q,
          "either": np.fmin(dd_k, dd_q), "both": np.fmax(dd_k, dd_q)}[cfg.idx_mode]

    # ── 시장 폭(breadth): 50일선 위 종목 비율 ────────────────────
    ma50 = close_panel.rolling(50, min_periods=40).mean()
    p50 = (close_panel > ma50).where(close_panel.notna() & ma50.notna()).mean(axis=1).values

    # ── 매크로: 공표 지연 반영 후 날짜 인덱스로 정렬 ─────────────
    def align(s: pd.Series, lag: int) -> np.ndarray:
        return s.reindex(s.index.union(d)).ffill().reindex(d).shift(lag).values

    fx20  = pd.Series(align(krw,  cfg.fx_lag),  index=d).pct_change(20).values
    r20   = pd.Series(align(us10, cfg.rate_lag), index=d).diff(20).values

    # ── 국면 판정 ─────────────────────────────────────────────────
    bull_raw = (
        _comb(up_k, up_q, cfg.idx_mode)
        & (p50 > cfg.b_hi)
        & (dd <= cfg.dmax)
        & ~np.where(np.isnan(fx20),  False, fx20  > cfg.fxmax)
        & ~np.where(np.isnan(r20),   False, r20   > cfg.rmax)
    )
    bear_raw = _comb(dn_k, dn_q, cfg.idx_mode) & (p50 < cfg.b_lo)

    bull = _persist(bull_raw, cfg.confirm)
    bear = _persist(bear_raw, cfg.confirm)

    state = np.where(bull, 2, np.where(bear, 0, 1)).astype(np.int8)

    return pd.DataFrame({
        "state": state, "p50": p50,
        "fx20": fx20, "r20": r20,
        "dd_k": dd_k, "dd_q": dd_q,
    }, index=d)
