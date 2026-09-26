"""청산 규칙 비교 엔진 — A안 체결 가정(결정일 다음 거래일 종가 매수, 다음 결정 다음 거래일 종가 매도).
변형(파라미터는 3차 연구에서 가져온 값, ③에 맞춰 튜닝한 것 아님):
  HOLD  : 다음 리밸런싱까지 보유
  T10   : 종목 트레일링 스톱 10% (②와 동일 폭) — 종가가 매수 후 고점 대비 −10% 이하인 날의 다음 거래일 종가 매도
  R200  : KOSPI 종가 < 200일 이동평균(3차 V6와 동일)인 날의 다음 거래일 종가 전량 매도, 다음 리밸런싱까지 현금
  T10+R200
청산 후 현금은 CD 금리. 거래비용 편도 50bp(research/variants COST와 동일), 계속 보유 종목은 거래 없음.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COST_BP = 50
EXIT_RULES = {"HOLD": {}, "T10": {"stop": 0.10}, "R200": {"ma": 200}, "T10+R200": {"stop": 0.10, "ma": 200}}


def entry_day(days: pd.DatetimeIndex, as_of) -> pd.Timestamp | None:
    nxt = days[days > pd.Timestamp(as_of)]
    return nxt[0] if len(nxt) else None


def simulate(book: list[dict], ra: pd.DataFrame, kospi: pd.Series, rf_m: pd.Series,
             capital: float, rule: dict) -> tuple[pd.Series, pd.DataFrame]:
    """book: [{as_of, tickers(list), gate(0/1)}] 결정 순. 반환: 일별 NAV, 청산 이벤트 로그."""
    days = ra.index
    risk = None
    if "ma" in rule:
        k = kospi.reindex(days).ffill()
        risk = (k < k.rolling(rule["ma"], min_periods=rule["ma"]).mean())
    book = sorted(book, key=lambda b: b["as_of"])
    starts = [entry_day(days, b["as_of"]) for b in book]
    nav, log = {}, []
    cash, pos = float(capital), {}                 # pos: ticker → [평가액, 매수후 고점 평가액]
    first = next((s for s in starts if s is not None), None)
    if first is None:
        return pd.Series(dtype=float), pd.DataFrame()
    pending_sell: set[str] = set()
    for i, d in enumerate(days[days >= first]):
        n_m = int((days.to_period("M") == d.to_period("M")).sum())
        cash *= 1 + float(rf_m.get(d.to_period("M"), 0.0)) / max(n_m, 1)
        for t in list(pos):                         # 1) 당일 수익 반영
            r = ra.at[d, t] if t in ra.columns else 0.0
            pos[t][0] *= 1 + (0.0 if pd.isna(r) else float(r))
            pos[t][1] = max(pos[t][1], pos[t][0])
        for t in list(pending_sell):                # 2) 전일 트리거 → 당일 종가 매도
            if t in pos:
                cash += pos.pop(t)[0] * (1 - COST_BP / 1e4)
        pending_sell.clear()
        if d in starts:                             # 3) 리밸런싱: 당일 종가에 교체
            b = book[starts.index(d)]
            tgt = set(b["tickers"]) if b.get("gate", 1) == 1 else set()
            if risk is not None and bool(risk.get(d, False)):
                tgt = set()                          # 왜: 위험 구간에 재진입하면 다음날 바로 청산되는 왕복 비용만 발생
            for t in [t for t in pos if t not in tgt]:
                cash += pos.pop(t)[0] * (1 - COST_BP / 1e4)
            new = [t for t in tgt if t not in pos]
            total = cash + sum(v[0] for v in pos.values())
            per = total / len(tgt) if tgt else 0.0
            for t in new:                           # 신규만 매수(계속 보유분은 무거래) — 현금 한도 내
                amt = min(per, cash)
                if amt <= 0:
                    break
                cash -= amt
                pos[t] = [amt * (1 - COST_BP / 1e4)] * 2
            for t in pos:                            # 왜: 트레일링 기준은 이번 보유 구간의 고점부터
                pos[t][1] = pos[t][0]
        else:                                       # 4) 청산 트리거 판정(당일 종가 기준 → 다음날 매도)
            if "stop" in rule:
                for t, (v, pk) in pos.items():
                    if v / pk - 1 <= -rule["stop"]:
                        pending_sell.add(t); log.append(dict(date=d, ticker=t, why=f"트레일링 {rule['stop']:.0%}"))
            if risk is not None and bool(risk.get(d, False)) and pos:
                pending_sell |= set(pos); log.append(dict(date=d, ticker="ALL", why=f"KOSPI<{rule['ma']}일선"))
        nav[d] = cash + sum(v[0] for v in pos.values())
    return pd.Series(nav), pd.DataFrame(log)


def stats(nav: pd.Series, capital: float) -> dict:
    if len(nav) < 2:
        return {}
    r = nav.pct_change().dropna(); yrs = len(nav) / 252
    dd = nav / nav.cummax() - 1
    return {"최종(만원)": round(nav.iloc[-1] / 1e4), "누적": f"{(nav.iloc[-1] / capital - 1) * 100:+.1f}%",
            "CAGR": f"{((nav.iloc[-1] / capital) ** (1 / yrs) - 1) * 100:+.1f}%" if yrs >= 1 else "1년 미만",
            "MDD": f"{dd.min() * 100:.1f}%", "Sharpe": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else np.nan}
