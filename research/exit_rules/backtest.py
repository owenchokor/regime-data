"""청산 규칙 비교 — ① 후보(52WH 상위20 × 월간 게이트) 과거 전 구간, 원금 1억, A안 체결(익일 종가).
③v2는 LLM 채점이라 과거 백테스트 불가(사후정보 누설) → ①로 규칙 자체의 성격만 본다. ③은 forward로만 판정.
실행: DATA_DIR=<release 폴더> python research/exit_rules/backtest.py  → research/exit_rules/out/"""
import os, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "forward"), str(ROOT / "research/gated_stocks"), str(ROOT / "research/variants"), str(ROOT / "research/monthly_timing")]
import core, stocks
DATA = os.environ.get("DATA_DIR", str(ROOT / "data")); core.DATA = stocks.DATA = DATA
from variants import Sim
from exit_sim import EXIT_RULES, simulate, stats
import run_forward as rf_mod

CAPITAL = 100_000_000
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
c, s, v, mc = stocks.load_panels()
ra, E = stocks.adjusted_returns(c, s)
MP = stocks.monthly_panel(ra, c, v, mc, E[E.type == "unmatched"])
cwd = os.getcwd(); os.chdir(OUT); pickle.dump({"MP": MP}, open("stocks.pkl", "wb")); sim = Sim(); os.remove("stocks.pkl"); os.chdir(cwd)
W = sim.select({})
gate, _ = rf_mod.kospi_gate()
rf = rf_mod.cd_rate(W.index.union(W.index + 1))
kospi = core.load_daily()["kospi"].dropna()
last_day = {p: d for p, d in pd.Series(ra.index, index=ra.index.to_period("M")).groupby(level=0).max().items()}
book = []
for t in W.index:
    if pd.isna(gate.get(t)) or t not in last_day:
        continue
    w = W.loc[t].dropna(); book.append(dict(as_of=last_day[t], tickers=list(w[w > 0].index), gate=int(gate[t])))
navs, logs = {}, {}
for nm, rule in EXIT_RULES.items():
    navs[nm], logs[nm] = simulate(book, ra, kospi, rf, CAPITAL, rule)
nav = pd.DataFrame(navs)
k = kospi.reindex(nav.index).ffill(); nav["KOSPI"] = CAPITAL * k / k.iloc[0]
T = pd.DataFrame({n: stats(nav[n], CAPITAL) for n in nav}).T
T["청산 이벤트"] = [len(logs.get(n, [])) if n in logs else "-" for n in T.index]
nav.to_parquet(OUT / "nav.parquet"); T.to_csv(OUT / "stats.csv")
print(f"기간 {nav.index[0].date()} ~ {nav.index[-1].date()}  결정 {len(book)}개월(게이트 ON {sum(b['gate'] for b in book)})")
print(T.to_string())

# 시각화: 누적(로그) + 낙폭
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
fp = "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf"
if Path(fp).exists():
    font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"] = False
BG, TX, DIM, GR = "#0b0f17", "#e5e7eb", "#8b95a7", "#1f2937"
COL = {"HOLD": "#22d3ee", "T10": "#e879f9", "R200": "#fbbf24", "T10+R200": "#a3e635", "KOSPI": "#9ca3af"}
LAB = {"HOLD": "보유→리밸런싱", "T10": "+트레일링 10%", "R200": "+KOSPI 200일선 이탈", "T10+R200": "+둘 다", "KOSPI": "KOSPI"}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), dpi=150, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.05), facecolor=BG)
for a in (a1, a2):
    a.set_facecolor(BG); a.tick_params(colors=DIM); a.grid(color=GR, lw=0.6)
    for sp in a.spines.values(): sp.set_color(GR)
for n in nav:
    a1.plot(nav.index, nav[n] / 1e8, color=COL[n], lw=2.2 if n != "KOSPI" else 1.4, ls="-" if n != "KOSPI" else "--",
            label=f"{LAB[n]}  {T.loc[n, '최종(만원)'] / 1e4:.2f}억 · CAGR {T.loc[n, 'CAGR']} · MDD {T.loc[n, 'MDD']}")
    a2.plot(nav.index, (nav[n] / nav[n].cummax() - 1) * 100, color=COL[n], lw=1.2, ls="-" if n != "KOSPI" else "--")
a1.set_yscale("log"); a1.axhline(1, color=DIM, lw=0.8)
from matplotlib.ticker import FuncFormatter, LogLocator
a1.yaxis.set_major_locator(LogLocator(base=2)); a1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}억")); a1.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
a1.set_ylabel("평가액 (억원, 로그)", color=DIM); a2.set_ylabel("낙폭 %", color=DIM)
a1.set_title("청산 규칙 비교 — ① 52WH 후보 20 × 월간 게이트, 원금 1억, 익일 종가 체결, 편도 50bp", color=TX, fontsize=13, loc="left")
leg = a1.legend(loc="upper left", fontsize=9, facecolor=BG, edgecolor=GR)
for t in leg.get_texts(): t.set_color(TX)
fig.savefig(OUT / "exit_rules.png", facecolor=BG, bbox_inches="tight")
print("saved", OUT / "exit_rules.png")
