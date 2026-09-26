"""③v2 월초 예측 적중 기록.
결정 D의 점수 vs 실현 수익: 진입 = 결정일 다음 거래일 종가(A안), 1M = 다음 리밸런싱(D+2월 첫 거래일) 종가, 3M = D+4월 첫 거래일 종가.
게이트 OFF 월도 규칙상 선정(shadow) 기준으로 평가 — 예측력 측정과 매수 여부를 분리.
비교군: 후보 20 평균, 52WH 규칙순위 상위 5(기계 기준), KOSPI.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from exit_sim import entry_day

BUCKETS = [(0, 30, "0–29 회피"), (30, 50, "30–49"), (50, 70, "50–69 관망"), (70, 85, "70–84 편입"), (85, 101, "85–100")]   # rubric.md 구간
HORIZONS = {"1m": 1, "3m": 3}


def load_book(dec: Path) -> list[dict]:
    """③v2 실운용 결정만(forward/decisions). 게이트 반영된 selected 로 매매."""
    out = []
    for f in sorted(dec.glob("*_scores.csv")):
        D = f.stem.replace("_scores", "")
        c = pd.read_csv(dec / f"{D}.csv", dtype={"ticker": str}); s = pd.read_csv(f, dtype={"ticker": str})
        if "selected" not in s:
            continue
        out.append(dict(month=D, as_of=pd.Timestamp(c.as_of.iloc[0]), gate=int(c.gate.iloc[0]),
                        tickers=list(s.ticker[s.selected == 1]), cand=c, scores=s))
    return out


def _ret(ra: pd.DataFrame, tk, a, b) -> pd.Series:
    D = ra.index[(ra.index > a) & (ra.index <= b)]
    return (1 + ra.loc[D, list(tk)].fillna(0.0)).prod() - 1


def predictions(book: list[dict], ra: pd.DataFrame, kospi: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days = ra.index; last = days[-1]; rows, mon = [], []
    k = kospi.reindex(days).ffill()
    for b in book:
        e0 = entry_day(days, b["as_of"])
        if e0 is None:
            continue
        s = b["scores"].merge(b["cand"][["ticker", "rule_rank"]], on="ticker", how="left", suffixes=("", "_c"))
        s["rule_rank"] = s["rule_rank"].fillna(s.get("rule_rank_c"))
        rec = s[["ticker", "name", "score", "g", "v", "m", "rule_rank", "shadow", "selected"]].copy()
        rec.insert(0, "month", b["month"]); rec["entry"] = e0.date()
        m = dict(month=b["month"], gate=b["gate"], entry=e0.date(), n_pick=int(s.shadow.sum()))
        for h, n in HORIZONS.items():
            p_exit = pd.Period(b["month"], "M") + 1 + n
            ex = days[days.to_period("M") == p_exit]
            done = len(ex) > 0
            e1 = ex[0] if done else last
            r = _ret(ra, s.ticker, e0, e1).reindex(s.ticker).values
            rec[f"ret_{h}"] = r; rec[f"done_{h}"] = int(done)
            kr = float(k.get(e1) / k.get(e0) - 1)
            ok = ~np.isnan(r)
            sel = s.shadow.values == 1
            top5 = s.rule_rank.values <= 5
            m.update({f"IC_{h}": pd.Series(s.score.values[ok]).corr(pd.Series(r[ok]), method="spearman") if ok.sum() > 3 else np.nan,
                      f"선정_{h}": np.nanmean(r[sel]) if sel.any() else np.nan, f"후보20_{h}": np.nanmean(r),
                      f"52WH상위5_{h}": np.nanmean(r[top5]), f"KOSPI_{h}": kr, f"상태_{h}": "확정" if done else f"진행중(~{last.date()})"})
        rows.append(rec); mon.append(m)
    P = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    M = pd.DataFrame(mon)
    B = pd.DataFrame()
    if len(P) and P.done_1m.any():
        q = P[P.done_1m == 1].copy()
        q["excess"] = q.ret_1m - q.groupby("month").ret_1m.transform("mean")      # 월내 후보 평균 대비 — 시장 등락 상쇄
        q["bucket"] = pd.cut(q.score, [a for a, _, _ in BUCKETS] + [101], right=False, labels=[l for _, _, l in BUCKETS])
        B = q.groupby("bucket", observed=False).agg(n=("ret_1m", "size"), 평균수익=("ret_1m", "mean"),
                                                     승률=("ret_1m", lambda x: (x > 0).mean()), 후보대비=("excess", "mean"))
    return P, M, B


def status_section(M: pd.DataFrame, B: pd.DataFrame) -> list[str]:
    if M.empty:
        return []
    L = ["\n## ③v2 예측 적중 기록 (월초 채점 vs 실현, 익일 종가 진입)",
         "IC = 점수와 실현수익의 순위상관(20종목). 선정 = 규칙상 선정(게이트 OFF 월 포함). 1M은 다음 리밸런싱까지.\n"]
    show = M[["month", "gate", "n_pick", "IC_1m", "선정_1m", "52WH상위5_1m", "후보20_1m", "KOSPI_1m", "상태_1m", "IC_3m", "선정_3m", "상태_3m"]].copy()
    for c in [c for c in show if any(x in c for x in ("선정", "상위5", "후보20", "KOSPI"))]:
        show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x * 100:+.1f}%")
    for c in ("IC_1m", "IC_3m"):
        show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:+.2f}")
    L.append(show.to_markdown(index=False))
    done = M[M["상태_1m"] == "확정"]
    if len(done):
        ic = done.IC_1m.dropna()
        se = ic.std(ddof=1) / np.sqrt(len(ic)) if len(ic) > 1 else np.nan
        win = (done["선정_1m"] > done["후보20_1m"]).mean()
        L.append(f"\n확정 {len(done)}개월 · 평균 IC(1M) {ic.mean():+.3f}" + (f" ± {se:.3f}" if pd.notna(se) else "") +
                 f" · 선정군이 후보20을 이긴 달 {win * 100:.0f}%")
    if len(B):
        L += ["\n### 점수 구간별 실현 (1M 확정분) — 기준점 재보정용", B.assign(
            평균수익=B.평균수익.map(lambda x: "" if pd.isna(x) else f"{x * 100:+.1f}%"), 승률=B.승률.map(lambda x: "" if pd.isna(x) else f"{x * 100:.0f}%"),
            후보대비=B.후보대비.map(lambda x: "" if pd.isna(x) else f"{x * 100:+.1f}%p")).to_markdown()]
        y = [0 if pd.isna(v) else round(v * 100, 1) for v in B.후보대비]
        lo, hi = min(y + [0]) - 1, max(y + [0]) + 1
        L += ["```mermaid", "xychart-beta", '  title "점수 구간별 후보 평균 대비 초과수익(%p, 1M)"',
              "  x-axis [" + ", ".join(f'"{b}"' for b in B.index) + "]", f'  y-axis "%p" {lo:.0f} --> {hi:.0f}',
              "  bar [" + ", ".join(str(v) for v in y) + "]", "```"]
    return L
