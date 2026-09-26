"""③v2 월말 결정 입력(decisions/{월}.csv) 생성.
후보 = ① 신호 20종목(52WH 순). 에이전트 채점 전에 기계적으로 계산 가능한 값만 담는다.
9차 IC 근거로 넣는 값: eps_g3(+0.12, 최강) / h52 정확도달·amt_ratio·r1m·vol20(일관 불리, −0.05~−0.08).
딜·재무위험 플래그는 DART 재수집·필터 사전등록 전이라 'pending'으로 둔다(임의 기준 금지).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

WIN_R1M, WIN_VOL, WIN_AMT_S, WIN_AMT_L = 21, 20, 20, 120   # research/monthbook/feats_ex.py 와 동일 창
EPS_LAG = 3                                                  # 9차 eps_g3 정의(3개월 전 대비)


def is_month_end(day: pd.Timestamp) -> bool:
    """day가 해당 월 KRX 마지막 거래일인지. 왜: A안은 '월말 저녁 채점 → 익일 종가 매수'라 월말 당일에 확정해야 함."""
    day = pd.Timestamp(day).normalize()
    try:
        import exchange_calendars as xc
        cal = xc.get_calendar("XKRX")
        if day <= cal.last_session:
            return cal.next_session(day).month != day.month
    except Exception as e:
        print("calendar fallback:", e)
    nxt = day + pd.offsets.BDay(1)            # 폴백: 주말만 고려(공휴일 미반영 — 로그로 표시)
    return nxt.month != day.month


def _fund(t: pd.Period, day: pd.Timestamp, data: str) -> tuple[pd.DataFrame, str]:
    """결정일 KRX 펀더멘털. pykrx 당일 조회 우선, 실패 시 fund_monthly 동월 스냅샷."""
    try:
        from pykrx import stock
        fn = getattr(stock, "get_market_fundamental_by_ticker", None) or stock.get_market_fundamental
        ds = day.strftime("%Y%m%d")
        parts = [fn(ds, market=m) for m in ("KOSPI", "KOSDAQ")]
        df = pd.concat([p for p in parts if not p.empty])
        if len(df):
            df.index = df.index.astype(str)
            return df, f"pykrx {ds}"
    except Exception as e:
        print("fund pykrx fail:", e)
    fm = pd.read_parquet(f"{data}/fund_monthly.parquet")
    fm = fm[pd.to_datetime(fm.date).dt.to_period("M") == t]
    if len(fm):
        return fm.drop_duplicates("ticker").set_index("ticker"), f"fund_monthly {fm.date.max()}"
    return pd.DataFrame(columns=["EPS", "BPS", "PER", "PBR", "DIV"]), "없음"


def build(t: pd.Period, sig: dict, ra: pd.DataFrame, day: pd.Timestamp, data: str, out: Path) -> Path:
    tk = [h["ticker"] for h in sig["holdings"]]
    D = ra.index[ra.index <= day]
    r1m = (1 + ra.loc[D[-WIN_R1M:], tk].fillna(0)).prod() - 1
    vol20 = ra.loc[D[-WIN_VOL:], tk].std() * np.sqrt(252)
    amt = pd.read_parquet(f"{data}/panel_amount.parquet"); amt = amt[amt.index <= day]
    amt_ratio = amt[tk].iloc[-WIN_AMT_S:].mean() / amt[tk].iloc[-WIN_AMT_L:].mean()

    F, fsrc = _fund(t, day, data)
    fm = pd.read_parquet(f"{data}/fund_monthly.parquet"); fm["p"] = pd.to_datetime(fm.date).dt.to_period("M")
    old = fm[fm.p == t - EPS_LAG].drop_duplicates("ticker").set_index("ticker")["EPS"]
    eps = F["EPS"].reindex(tk).astype(float) if "EPS" in F else pd.Series(np.nan, index=tk)
    eps_old = old.reindex(tk).astype(float)
    eps_g3 = (eps - eps_old) / eps_old.abs().replace(0, np.nan)

    sec = pd.read_parquet(f"{data}/sector_monthly.parquet"); sec = sec[sec.date == sec.date.max()].set_index("ticker")
    df = pd.DataFrame({
        "decision_month": str(t), "as_of": str(day.date()),
        "ticker": tk,
        "name": [h["name"] or sec["종목명"].get(h["ticker"], "") for h in sig["holdings"]],
        "sector": sec["업종명"].reindex(tk).fillna("미분류").values,
        "rule_rank": range(1, len(tk) + 1),
        "high52": [h["high52"] for h in sig["holdings"]],
        "at_52wh": [int(h["high52"] >= 0.9999) for h in sig["holdings"]],
        "mcap_rank": [h["mcap_rank"] for h in sig["holdings"]],
        "r1m": r1m.values, "vol20": vol20.values, "amt_ratio": amt_ratio.values,
        "eps": eps.values, "eps_g3": eps_g3.values,
        # 왜: KRX EPS는 연 1회(5월 사업보고서 반영) 갱신 → 그 외 달 eps_g3=0은 '성장 없음'이 아니라 '정보 없음'
        "eps_updated_3m": ((eps != eps_old) & eps.notna() & eps_old.notna()).astype(int).values,
        "per": F["PER"].reindex(tk).values if "PER" in F else np.nan,
        "pbr": F["PBR"].reindex(tk).values if "PBR" in F else np.nan,
        "div": F["DIV"].reindex(tk).values if "DIV" in F else np.nan,
        "loss": (eps <= 0).astype(float).where(eps.notna()).values,
        "impair": (F["BPS"].reindex(tk).astype(float) <= 0).astype(float).values if "BPS" in F else np.nan,
        "deal_flag": "pending", "fin_flag": "pending",   # 왜: DART 필터 사전등록 전 — 기준 확정 후 채움
        "gate": sig["gate"], "fund_src": fsrc,
    })
    (out / "decisions").mkdir(exist_ok=True)
    f = out / "decisions" / f"{t}.csv"
    df.round(5).to_csv(f, index=False)
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as fh:
            fh.write(f"decision={t}\n")
    print("decision csv", f, "fund", fsrc)
    return f
