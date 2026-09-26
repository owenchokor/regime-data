"""3차: 52WH × KOSPI 게이트 변형 선별(A). 변형·파라미터·통과기준은 실행 전 고정.

통과 기준 (WF-OOS 2019-01~2024-06, 50bp, 초과수익 Sharpe):
  (1) Sharpe ≥ 기준형 − 0.05
  (2) MDD 또는 CVaR5 가 기준형보다 3%p 이상 개선
  (3) 파라미터 이웃값(스톱 폭 등)에서 (1)이 유지
holdout(2024-07~)은 이미 관측된 구간이라 판정에 쓰지 않고 진단으로만 표시.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from stocks import load_panels, adjusted_returns
import pickle

BASE_TOPN, BASE_UNIV, COST = 20, 500, 50
VARIANTS = {
    "BASE": {},
    "V1 과열제외(1M 상위10%)": {"exclude_hot": 0.10},
    "V2 역변동성 가중": {"invvol": True},
    "V3 회전율 증가 확인": {"turn_up": True},
    "V4 종목 트레일링 스톱 15%": {"stock_stop": 0.15},
    "V5 포트폴리오 스톱 10%": {"port_stop": 0.10},
    "V6 일간 게이트(200일선)": {"daily_gate": 200},
}
NEIGHBORS = {  # (3) 이웃값 점검용
    "V1 과열제외(1M 상위10%)": [{"exclude_hot": x} for x in (0.05, 0.20)],
    "V4 종목 트레일링 스톱 15%": [{"stock_stop": x} for x in (0.10, 0.20)],
    "V5 포트폴리오 스톱 10%": [{"port_stop": x} for x in (0.07, 0.15)],
    "V6 일간 게이트(200일선)": [{"daily_gate": x} for x in (120, 250)],
}


class Sim:
    def __init__(self):
        c, s, v, mc = load_panels()
        ra, _ = adjusted_returns(c, s)
        self.ra = ra.fillna(0.0)                          # 상폐 후·정지일 0 → 마지막 가격 동결
        self.alive = c.notna()
        turn = (v / s).replace([np.inf, -np.inf], np.nan)
        self.turn_ratio = (turn.rolling(20, min_periods=15).mean() /
                           turn.rolling(120, min_periods=90).mean())
        self.per = c.index.to_period("M")
        S = pickle.load(open("stocks.pkl", "rb"))["MP"]
        self.S = S
        from core import load_daily
        kd = load_daily()["kospi"]
        self.kospi_d = kd

    def select(self, opt: dict, univ=BASE_UNIV, topn=BASE_TOPN) -> pd.DataFrame:
        S = self.S; F = S["feats"]
        U = S["elig"] & (S["rank"] <= univ)
        if "exclude_hot" in opt:
            hot = F["mom1"].where(U).rank(axis=1, pct=True) > 1 - opt["exclude_hot"]
            U = U & ~hot
        if opt.get("turn_up"):
            tr = self.turn_ratio.groupby(self.per).last().reindex(U.index)
            U = U & (tr > 1.0)
        sc = F["high52"].where(U & F["high52"].notna())
        rr = sc.rank(axis=1, ascending=False, method="first")
        sel = rr <= topn
        if opt.get("invvol"):
            w = (1 / F["vol12"]).where(sel)
        else:
            w = sel.astype(float).where(sel)
        return w.div(w.sum(axis=1), axis=0)

    def month_returns(self, W: pd.DataFrame, opt: dict, gate_m: pd.Series, rf: pd.Series):
        """결정월 t의 W → t+1월 일별 경로로 스톱 반영. 반환: (수익월 인덱스 Series, 스톱 발생 비중)."""
        out, stopped = {}, {}
        ra = self.ra
        dg = None
        if "daily_gate" in opt:
            k = self.kospi_d
            dg = (k < k.rolling(opt["daily_gate"]).mean())      # True = 위험 → 청산
        days_by_m = {p: ra.index[self.per == p] for p in pd.unique(self.per)}
        prevW = None
        for t in W.index:
            p1 = t + 1
            if p1 not in days_by_m or pd.isna(gate_m.get(t, np.nan)):
                continue
            w0 = W.loc[t].dropna(); w0 = w0[w0 > 0]
            g = float(gate_m[t])
            rfm = float(rf.get(p1, 0.0))
            if len(w0) == 0 or g == 0:
                out[p1] = rfm; stopped[p1] = 0.0; prevW = None; continue
            D = days_by_m[p1]
            R = ra.loc[D, w0.index].values                      # 일×종목
            cum = np.cumprod(1 + R, axis=0)
            active = np.ones(R.shape, bool)
            ex_day = np.full(R.shape[1], R.shape[0])            # 청산 적용 시작일(그날부터 현금)
            if "stock_stop" in opt:
                peak = np.maximum.accumulate(np.vstack([np.ones(R.shape[1]), cum]), axis=0)[1:]
                hit = cum / peak - 1 <= -opt["stock_stop"]
                for j in range(R.shape[1]):
                    idx = np.where(hit[:, j])[0]
                    if len(idx):
                        ex_day[j] = min(ex_day[j], idx[0] + 2)   # 다음날 종가 체결 → 그 다음날부터 무보유
            if "port_stop" in opt or dg is not None:
                pv = (cum * w0.values).sum(axis=1)
                trig = np.zeros(R.shape[0], bool)
                if "port_stop" in opt:
                    pk = np.maximum.accumulate(np.r_[1.0, pv])[1:]
                    trig |= pv / pk - 1 <= -opt["port_stop"]
                if dg is not None:
                    trig |= dg.reindex(D).fillna(False).values
                idx = np.where(trig)[0]
                if len(idx):
                    ex_day[:] = np.minimum(ex_day, idx[0] + 2)
            for j in range(R.shape[1]):
                active[ex_day[j]:, j] = False
            Ra = np.where(active, R, 0.0)
            val = np.cumprod(1 + Ra, axis=0)[-1]                 # 종목별 월말 가치
            frac_cash_days = 1 - active.mean(axis=0)
            gross = float((w0.values * (val - 1)).sum() + (w0.values * frac_cash_days).sum() * rfm)
            ex_frac = float(w0.values[ex_day < R.shape[0]].sum())
            # 재선정 종목은 조기 청산분을 다시 사야 하므로 추가 비용
            nxt = W.loc[p1].dropna() if p1 in W.index else pd.Series(dtype=float)
            resel = np.isin(w0.index, nxt[nxt > 0].index) & (ex_day < R.shape[0])
            extra = 2 * COST / 1e4 * float(w0.values[resel].sum())
            out[p1] = g * (gross - extra) + (1 - g) * rfm
            stopped[p1] = ex_frac
        return pd.Series(out), pd.Series(stopped)


def turnover_cost(W: pd.DataFrame, gate_m: pd.Series) -> pd.Series:
    Wg = W.fillna(0).mul(gate_m.reindex(W.index).fillna(0), axis=0)
    tr = (Wg - Wg.shift(1).fillna(0)).abs().sum(axis=1)
    return (tr * COST / 1e4).shift(1)                           # 수익월 인덱스


# ── 데이터 추가 후 변형 (실행 전 고정) ──
DATA_VARIANTS = {
    "V7 업종 상한(업종당 최대 4)": {"sector_cap": 4},
    "V8 유동성 하위30% 제외": {"liq_cut": 0.30},
}
DATA_NEIGHBORS = {
    "V7 업종 상한(업종당 최대 4)": [{"sector_cap": x} for x in (3, 6)],
    "V8 유동성 하위30% 제외": [{"liq_cut": x} for x in (0.2, 0.4)],
}


def load_sector() -> pd.DataFrame:
    s = pd.read_parquet("../../data/sector_monthly.parquet")
    s["p"] = pd.PeriodIndex(pd.to_datetime(s["date"]), freq="M")
    return s.pivot_table(index="p", columns="ticker", values="업종명", aggfunc="first")


def select_sector_cap(sim: "Sim", cap: int, sector: pd.DataFrame, univ=BASE_UNIV, topn=BASE_TOPN):
    S = sim.S; F = S["feats"]
    U = S["elig"] & (S["rank"] <= univ)
    sc = F["high52"].where(U & F["high52"].notna())
    W = pd.DataFrame(np.nan, index=sc.index, columns=sc.columns)
    for t in sc.index:
        row = sc.loc[t].dropna().sort_values(ascending=False)
        if row.empty:
            continue
        sec = sector.loc[t].reindex(row.index) if t in sector.index else pd.Series(index=row.index, dtype=object)
        cnt, pick = {}, []
        for tk in row.index:
            g = sec.get(tk)
            g = "NA" if pd.isna(g) else g
            if cnt.get(g, 0) < cap:
                pick.append(tk); cnt[g] = cnt.get(g, 0) + 1
            if len(pick) == topn:
                break
        W.loc[t, pick] = 1.0 / len(pick)
    return W


def select_liq(sim: "Sim", cut: float, amount: pd.DataFrame, univ=BASE_UNIV, topn=BASE_TOPN):
    """형성월 마지막 20거래일 평균 거래대금이 유니버스 내 하위 cut 이면 제외."""
    S = sim.S; F = S["feats"]
    U = S["elig"] & (S["rank"] <= univ)
    a20 = amount.rolling(20, min_periods=15).mean().groupby(amount.index.to_period("M")).last()
    a20 = a20.reindex(index=U.index, columns=U.columns)
    low = a20.where(U).rank(axis=1, pct=True) <= cut
    U = U & ~low & a20.notna()
    sc = F["high52"].where(U & F["high52"].notna())
    sel = sc.rank(axis=1, ascending=False, method="first") <= topn
    w = sel.astype(float).where(sel)
    return w.div(w.sum(axis=1), axis=0)
