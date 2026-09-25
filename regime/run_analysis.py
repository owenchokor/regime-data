"""
국면 판정 전체 실행: 패널 전처리 → 지표 → 그리드(IS) → OOS 검증 → 백테스트 → 차트
입력: data/panel_{close,mcap,shares}.parquet (미수정), data/{kospi,kosdaq}.parquet, data/fred_*.parquet
실행: python -m regime.run_analysis
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

D, R = Path("data"), Path("results"); R.mkdir(exist_ok=True)
CFG = dict(
    IS_END="2020-12-31", FWD=20, BETA=0.5, MIN_COVER=0.15,
    MCAP_MIN=1.5e11,          # 기존 유니버스와 동일 (1,500억)
    JUMP=0.31,                # 가격제한폭(±30%) 초과 = 기업행위 추정 → 수익률 0 처리
    FX_LAG=5, RATE_LAG=1,     # 공표 지연 (FX는 확인 필요)
    GRID=dict(ma_n=[50,120,200], b_hi=[.5,.55,.6,.65], b_lo=[.3,.35,.4,.45],
              dmax=[4,6,99], fxmax=[np.inf,.03,.05], rmax=[np.inf,.3,.5],
              idx_mode=["kospi","kosdaq","either","both"], confirm=[1,3,5]),  # 전부 임의값
)
BG, PAPER = "#070b14", "#05080f"; NEON = ["#00f0ff","#ff2bd6","#39ff14","#ffe600","#ff6b00"]
def _layout(fig, title, h=500):
    fig.update_layout(template="plotly_dark", title=title, height=h, plot_bgcolor=BG,
                      paper_bgcolor=PAPER, font=dict(color="#d0d6e0")); return fig

# ── 1. 패널 전처리 ──────────────────────────────────────────────
close = pd.read_parquet(D/"panel_close.parquet"); mcap = pd.read_parquet(D/"panel_mcap.parquet")
close.index = pd.to_datetime(close.index); mcap.index = pd.to_datetime(mcap.index)
common = [c for c in close.columns if str(c).endswith("0")]   # 우선주 제외 (보통주 코드 끝자리 0 관행)
close, mcap = close[common], mcap[common]
close = close.where(close > 0)
ret = close.pct_change(fill_method=None)
jump = ret.abs() > CFG["JUMP"]
print("가격제한폭 초과 점프(0 처리):", int(jump.sum().sum()))
ret = ret.mask(jump, 0.0)                                      # 왜: 미수정 패널의 분할·감자 허구 수익 제거
adj = (1 + ret.fillna(0)).cumprod().where(close.notna())       # 의사 수정종가
member = (mcap >= CFG["MCAP_MIN"]) & close.notna()
d = adj.index; is_ = (d <= CFG["IS_END"])
print(f"기간 {d[0].date()}~{d[-1].date()}, 일평균 유니버스 {member.sum(1).mean():.0f}종목")

# ── 2. 정답: 향후 FWD일 유니버스 개별주 수익률 중앙값 ─────────────
fwd = (adj.shift(-CFG["FWD"]) / adj - 1).where(member)
tgt = fwd.median(axis=1); ok = tgt.notna().values
q_lo, q_hi = tgt[is_ & ok].quantile([1/3, 2/3])
tb, tw = (tgt >= q_hi).values, (tgt <= q_lo).values

# ── 3. 관측 지표 (T 시점) ─────────────────────────────────────
ma50 = adj.rolling(50, min_periods=40).mean()
p50 = (adj > ma50).where(member & ma50.notna()).astype(float).mean(axis=1).values
kospi, kosdaq = (pd.read_parquet(D/f"{n}.parquet") for n in ("kospi","kosdaq"))
ci = {"kospi": kospi["종가"].reindex(d), "kosdaq": kosdaq["종가"].reindex(d)}
cv = {"kospi": kospi["거래량"].reindex(d), "kosdaq": kosdaq["거래량"].reindex(d)}
ddm = {k: ((ci[k].pct_change() <= -0.002) & (cv[k] > cv[k].shift(1))).rolling(25, min_periods=1).sum().values for k in ci}
def macro(sid, lag):
    s = pd.read_parquet(D/f"fred_{sid}.parquet").iloc[:, 0]
    return s.reindex(s.index.union(d)).ffill().reindex(d).shift(lag)
fx20 = macro("DEXKOUS", CFG["FX_LAG"]).pct_change(20).values
r20 = macro("DGS10", CFG["RATE_LAG"]).diff(20).values
G = CFG["GRID"]
up = {(k,n): (ci[k] > ci[k].rolling(n).mean()).values for k in ci for n in G["ma_n"]}
dn = {(k,n): (ci[k] < ci[k].rolling(n).mean()).values for k in ci for n in G["ma_n"]}

def comb(a, b, m): return {"kospi": a, "kosdaq": b, "either": a | b, "both": a & b}[m]
def ddsel(m):
    a, b = ddm["kospi"], ddm["kosdaq"]
    return {"kospi": a, "kosdaq": b, "either": np.fmin(a, b), "both": np.fmax(a, b)}[m]
def persist(x, c):  # 왜: 하루짜리 플립으로 잦은 진입·이탈 방지
    return x if c == 1 else np.convolve(x.astype(int), np.ones(c), "full")[:len(x)] == c
def regime(p):
    bull = (comb(up["kospi",p["ma_n"]], up["kosdaq",p["ma_n"]], p["idx_mode"]) & (p50 > p["b_hi"])
            & (ddsel(p["idx_mode"]) <= p["dmax"]) & ~(fx20 > p["fxmax"]) & ~(r20 > p["rmax"]))
    bear = comb(dn["kospi",p["ma_n"]], dn["kosdaq",p["ma_n"]], p["idx_mode"]) & (p50 < p["b_lo"])
    bull, bear = persist(bull, p["confirm"]), persist(bear, p["confirm"])
    return np.where(bull, 2, np.where(bear, 0, 1))
def score(st, m):
    b, t = (st == 2)[m], tb[m]
    if b.mean() < CFG["MIN_COVER"] or b.sum() == 0: return None
    ppv, rec = (b & t).sum() / b.sum(), (b & t).sum() / t.sum(); B2 = CFG["BETA"]**2
    w = (st == 0)[m]
    return dict(fb=(1+B2)*ppv*rec/(B2*ppv+rec) if ppv+rec else 0, ppv=ppv, rec=rec,
                cover=b.mean(), bear_ppv=(w & tw[m]).sum()/max(w.sum(),1), bear_cover=w.mean())

# ── 4. 그리드 (선택은 IS만) ───────────────────────────────────
rows, keys = [], list(G)
for vals in itertools.product(*G.values()):
    p = dict(zip(keys, vals)); st = regime(p)
    a, b = score(st, is_ & ok), score(st, ~is_ & ok)
    if a and b: rows.append({**p, **{f"is_{k}": v for k,v in a.items()}, **{f"oos_{k}": v for k,v in b.items()}})
grid = pd.DataFrame(rows).sort_values("is_fb", ascending=False); grid.to_csv(R/"grid.csv", index=False)
base = tb[~is_ & ok].mean()
print(f"\n유효 조합 {len(grid)} | 강세 기저율 IS {tb[is_&ok].mean():.3f} OOS {base:.3f}")
cols = keys + ["is_fb","is_ppv","is_cover","oos_fb","oos_ppv","oos_rec","oos_cover","oos_bear_ppv"]
print(grid.head(10)[cols].round(3).to_string(index=False))
print("IS-OOS fb 순위상관(Spearman):", round(grid["is_fb"].corr(grid["oos_fb"], method="spearman"), 3))
best = grid.iloc[0][keys].to_dict(); st = regime(best)

# ── 5. 백테스트: 강세일에만 유니버스 동일가중 보유 ──────────────
ew = ret.where(member.shift(1)).mean(axis=1)                   # T-1 소속 종목의 T 수익
pos = pd.Series((st == 2).astype(float), index=d).shift(1)
strat, bh = ew * pos, ci["kospi"].pct_change()
def perf(r, name):
    r = r.fillna(0); eq = (1 + r).cumprod(); yrs = len(r)/252
    live = r[pos.reindex(r.index).fillna(0) > 0] if name == "강세만" else r[r != 0]
    w, l = live[live > 0], live[live < 0]
    return dict(전략=name, 보유일=len(live), CAGR=eq.iloc[-1]**(1/yrs)-1,
                MDD=(eq/eq.cummax()-1).min(), 일승률=len(w)/max(len(live),1), 손익비=w.mean()/-l.mean())
rep = pd.DataFrame([{**perf(s[m], n), "구간": per} for m, per in [(is_,"IS"),(~is_,"OOS")]
                    for s, n in [(strat,"강세만"),(ew,"EW보유"),(bh,"KOSPI")]])
rep.to_csv(R/"perf.csv", index=False); print("\n", rep.round(3).to_string(index=False))
by = pd.DataFrame({"state": st, "tgt": tgt.values, "구간": np.where(is_,"IS","OOS")}).dropna()
tbl = by.groupby(["구간","state"])["tgt"].agg(["count","mean","median"]); print("\n국면별 향후20일 개별주 중앙값 수익:\n", tbl.round(4))
tbl.to_csv(R/"regime_fwd.csv")
out = pd.DataFrame({"state": st, "p50": p50, "fx20": fx20, "r20": r20, "dd_k": ddm["kospi"], "dd_q": ddm["kosdaq"]}, index=d)
out.to_parquet(R/"regime_daily.parquet")
(R/"best_params.json").write_text(json.dumps({**{k:(None if v==np.inf else v) for k,v in best.items()},
    "q_lo":q_lo,"q_hi":q_hi,"cfg":{k:v for k,v in CFG.items() if k!="GRID"}}, default=str, ensure_ascii=False, indent=1))
print("\n현재 국면:", {2:"강세",1:"박스권",0:"약세"}[int(st[-1])], d[-1].date())

# ── 6. 시각화 ────────────────────────────────────────────────
lab, col = {2:"강세",1:"박스권",0:"약세"}, {2:NEON[2],1:NEON[3],0:NEON[1]}
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.35,.15,.15,.35],
                    subplot_titles=["KOSPI + 국면","50일선 위 종목비율","분배일(KOSPI, 25일)","누적수익 (log)"])
for s in [2,1,0]:
    m = st == s
    fig.add_trace(go.Scatter(x=d[m], y=ci["kospi"].values[m], mode="markers", marker=dict(size=3,color=col[s]), name=lab[s]), 1, 1)
fig.add_trace(go.Scatter(x=d, y=p50, line=dict(color=NEON[0],width=1), name="p50"), 2, 1)
for y, c in [(best["b_hi"],NEON[2]),(best["b_lo"],NEON[1])]: fig.add_hline(y=y, line_dash="dot", line_color=c, row=2, col=1)
fig.add_trace(go.Bar(x=d, y=ddm["kospi"], marker_color=NEON[4], name="분배일"), 3, 1)
for s, n, c in [(strat,"강세만",NEON[2]),(ew,"EW보유",NEON[4]),(bh,"KOSPI",NEON[0])]:
    fig.add_trace(go.Scatter(x=d, y=(1+s.fillna(0)).cumprod(), name=n, line=dict(color=c)), 4, 1)
fig.add_vline(x=pd.Timestamp(CFG["IS_END"]), line_dash="dash", line_color="#888")
fig.update_yaxes(type="log", row=1, col=1); fig.update_yaxes(type="log", row=4, col=1)
_layout(fig, f"시장 국면 판정 best={best}", 1000).write_html(R/"regime_chart.html", include_plotlyjs="cdn")
top = grid.head(200)
_layout(go.Figure(go.Scatter(x=top["is_fb"], y=top["oos_fb"], mode="markers", marker=dict(color=NEON[0], size=5),
        text=top[keys].astype(str).agg(" ".join, axis=1))), "IS vs OOS Fβ (상위 200, 과적합 점검)", 500
        ).write_html(R/"grid_isoos.html", include_plotlyjs="cdn")
print("saved results/")
