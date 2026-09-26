"""Forward 모의운용 (사전등록, 2026-09-26 확정).
 ① BASE   : 52WH 상위20 동일가중 × KOSPI PxMA10 게이트, 월간 교체, 편도 50bp
 ② STOP10 : ①+ 종목 트레일링 스톱 10% (트리거 다음날 종가 체결) — 3차 이웃값에서 사후 발견된 신규가설
첫 결정월 2026-09(월말 신호) → 첫 보유월 2026-10. 판정·코드는 백테스트(research/variants)와 동일 경로 사용.
매일 실행: 패널 갱신 후 (1) 확정 월 신호 저장 (2) 진행 중 월 잠정 신호 (3) ② 스톱 모니터 (4) 완료 월 성과.
"""
from __future__ import annotations
import json, os, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research/gated_stocks"), str(ROOT / "research/variants"), str(ROOT / "research/monthly_timing")]
import core, stocks
DATA = os.environ.get("DATA_DIR", str(ROOT / "data"))
core.DATA = stocks.DATA = DATA
import variants
from variants import Sim, turnover_cost

FIRST_DECISION = pd.Period(os.environ.get("FWD_FIRST_DECISION", "2026-09"), "M")
STRATS = {"①BASE": {}, "②STOP10": {"stock_stop": 0.10}}
STOP = 0.10
OUT = ROOT / "forward"; (OUT / "signals").mkdir(parents=True, exist_ok=True)


def kospi_gate() -> tuple[pd.Series, pd.DataFrame]:
    k = core.load_daily()["kospi"].dropna()
    m = k.groupby(k.index.to_period("M")).last()
    ma = m.rolling(10).mean()
    g = (np.log(m / ma) > 0).astype(float).where(ma.notna())      # core.features pma10 과 동일 정의
    return g, pd.DataFrame({"kospi": m, "ma10": ma})


def cd_rate(idx: pd.PeriodIndex) -> pd.Series:
    ec = pd.read_parquet(f"{DATA}/ecos_721Y001.parquet")
    ec["p"] = pd.PeriodIndex(ec["time"], freq="M")
    cd = ec[ec.item_name == "CD(91일)"].set_index("p")["value"].sort_index()
    return (cd / 100 / 12).reindex(idx.union(cd.index)).ffill().reindex(idx)   # 왜: ECOS 미갱신월은 직전값(한계)


def names_for(tks) -> dict:
    try:
        from pykrx import stock
        return {t: stock.get_market_ticker_name(t) for t in tks}
    except Exception:
        return {}


def main() -> None:
    c, s, v, mc = stocks.load_panels()
    last_day = c.index.max(); cur = last_day.to_period("M")
    ra, E = stocks.adjusted_returns(c, s)
    MP = stocks.monthly_panel(ra, c, v, mc, E[E.type == "unmatched"])
    os.chdir(OUT); pickle.dump({"MP": MP}, open("stocks.pkl", "wb"))   # Sim 이 cwd 의 stocks.pkl 을 읽음
    sim = Sim(); os.remove("stocks.pkl")
    W = sim.select({})
    gate, km = kospi_gate()
    per = W.index
    rf = cd_rate(per.union(per + 1))

    # (1)(2) 신호: 확정 월 = 다음 달 데이터가 존재하는 월, 진행 중 월 = 잠정
    status_sig = {}
    for t in [p for p in per if p >= FIRST_DECISION]:
        final = t < cur
        f = OUT / "signals" / f"{t}.json"
        if final and f.exists():
            status_sig[t] = json.loads(f.read_text()); continue
        w = W.loc[t].dropna(); w = w[w > 0]
        w = w.loc[sim.S["feats"]["high52"].loc[t, w.index].sort_values(ascending=False).index]
        nm = names_for(w.index)
        sig = dict(decision_month=str(t), final=final, as_of=str(c.index[c.index.to_period("M") == t].max().date()),
                   gate=None if pd.isna(gate.get(t)) else int(gate[t]),
                   kospi=float(km.kospi.get(t, np.nan)), kospi_ma10=float(km.ma10.get(t, np.nan)),
                   holdings=[dict(ticker=k, name=nm.get(k, ""), weight=round(float(w[k]), 4),
                                  high52=round(float(sim.S["feats"]["high52"].loc[t, k]), 4),
                                  mcap_rank=int(sim.S["rank"].loc[t, k])) for k in w.index])
        if final: f.write_text(json.dumps(sig, ensure_ascii=False, indent=1))
        status_sig[t] = sig

    # (4) 완료 보유월 성과: 백테스트와 동일 함수
    rows = {}
    for nm_, opt in STRATS.items():
        mr, st = sim.month_returns(W, opt, gate, rf)
        x = mr - turnover_cost(W, gate).reindex(mr.index).fillna(0)
        rows[nm_] = x; rows[f"{nm_}_stopfrac"] = st
    perf = pd.DataFrame(rows)
    kr = km.kospi.pct_change()
    perf["KOSPI"] = kr.reindex(perf.index); perf["CD"] = rf.reindex(perf.index)
    perf = perf[(perf.index > FIRST_DECISION) & (perf.index < cur)]
    perf.index = perf.index.astype(str); perf.round(5).to_csv(OUT / "performance.csv", index_label="hold_month")

    # (3) ② 스톱 모니터: 진행 중 보유월 cur, 결정월 cur-1
    mon = []
    t = cur - 1
    if t in status_sig and status_sig[t]["gate"] == 1:
        D = ra.index[ra.index.to_period("M") == cur]
        for h in status_sig[t]["holdings"]:
            rr = ra.loc[D, h["ticker"]].fillna(0.0).values
            cum = np.cumprod(1 + rr); pk = np.maximum.accumulate(np.r_[1.0, cum])[1:]
            hit = np.where(cum / pk - 1 <= -STOP)[0]
            mon.append(dict(ticker=h["ticker"], name=h["name"], ret=cum[-1] - 1 if len(cum) else 0.0,
                            dd=(cum / pk - 1)[-1] if len(cum) else 0.0,
                            trigger=str(D[hit[0]].date()) if len(hit) else ""))
    write_status(last_day, cur, status_sig, perf, mon)
    print("ok", last_day.date(), "signals", [str(k) for k in status_sig], "perf rows", len(perf))


def write_status(last_day, cur, sig, perf, mon) -> None:
    L = [f"# Forward 모의운용 현황", f"갱신: 데이터 {last_day.date()} 기준\n",
         "사전등록: ① 52WH×PxMA10 기준형 / ② ① + 종목 트레일링 스톱 10%. 첫 보유월 2026-10.",
         "체결 가정: 백테스트와 동일(결정월 말 종가). 실제 체결은 신호 확인 다음 거래일이라 괴리 존재.\n"]
    if cur in sig:
        s = sig[cur]
        L += [f"## 잠정 신호 ({cur}, {s['as_of']} 기준 — 이 날이 월말이면 확정 신호와 동일)",
              f"게이트 {'ON(투자)' if s['gate'] == 1 else 'OFF(현금)'} · KOSPI {s['kospi']:.2f} vs MA10 {s['kospi_ma10']:.2f}\n",
              "| # | 티커 | 종목 | 52WH비 | 시총순위 |", "|---|---|---|---|---|"]
        L += [f"| {i+1} | {h['ticker']} | {h['name']} | {h['high52']:.3f} | {h['mcap_rank']} |" for i, h in enumerate(s["holdings"])]
    if mon:
        L += [f"\n## ② 스톱 모니터 ({cur} 보유분, 트리거 시 다음날 종가 청산)", "| 티커 | 종목 | 월중 수익 | 고점대비 | 트리거일 |", "|---|---|---|---|---|"]
        L += [f"| {m['ticker']} | {m['name']} | {m['ret']*100:+.1f}% | {m['dd']*100:.1f}% | {m['trigger'] or '-'} |" for m in mon]
    if len(perf):
        cum = (1 + perf[["①BASE", "②STOP10", "KOSPI"]]).prod() - 1
        L += ["\n## 완료 월 성과", perf.round(4).to_markdown(), "\n누적: " + " · ".join(f"{k} {v*100:+.1f}%" for k, v in cum.items())]
    (OUT / "STATUS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
