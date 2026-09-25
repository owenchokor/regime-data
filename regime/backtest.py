"""
백테스트 — 강세 국면에만 동일가중 보유

실행: python regime/backtest.py --close data/close.parquet
                                 --kospi data/kospi.parquet
                                 --kosdaq data/kosdaq.parquet
                                 --krw data/fred_DEXKOUS.parquet
                                 --us10 data/fred_DGS10.parquet
                                 --params results/best_params.json
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from regime.core import RegimeCFG, build_regime

IS_END = "2020-12-31"
BG, PAPER = "#070b14", "#05080f"
NEON = ["#00f0ff", "#ff2bd6", "#39ff14", "#ffe600", "#ff6b00"]

def _layout(fig, title, h=500):
    fig.update_layout(template="plotly_dark", title=title, height=h,
                      plot_bgcolor=BG, paper_bgcolor=PAPER, font=dict(color="#d0d6e0"))
    return fig

def perf(r: pd.Series, name: str) -> dict:
    r = r.fillna(0); live = r[r != 0]
    eq = (1 + r).cumprod()
    w = live[live > 0]; l = live[live < 0]
    yrs = max(len(r) / 252, 1e-6)
    return dict(name=name,
                n_days=int((r != 0).sum()),
                CAGR=round(eq.iloc[-1] ** (1 / yrs) - 1, 4),
                MDD=round((eq / eq.cummax() - 1).min(), 4),
                win=round(len(w) / max(len(live), 1), 4),
                payoff=round(w.mean() / -l.mean(), 4) if len(l) else float("nan"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--close",  required=True)
    ap.add_argument("--kospi",  required=True)
    ap.add_argument("--kosdaq", required=True)
    ap.add_argument("--krw",    required=True)
    ap.add_argument("--us10",   required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out",    default="results")
    args = ap.parse_args()

    params = json.loads(Path(args.params).read_text())
    cfg = RegimeCFG(**{k: v for k, v in params.items()
                       if k in RegimeCFG.__dataclass_fields__})

    close  = pd.read_parquet(args.close);  close.index  = pd.to_datetime(close.index)
    kospi  = pd.read_parquet(args.kospi);  kospi.index  = pd.to_datetime(kospi.index)
    kosdaq = pd.read_parquet(args.kosdaq); kosdaq.index = pd.to_datetime(kosdaq.index)
    krw    = pd.read_parquet(args.krw).iloc[:, 0]
    us10   = pd.read_parquet(args.us10).iloc[:, 0]

    res = build_regime(close, kospi, kosdaq, krw, us10, cfg)
    state = res["state"].values
    d = close.index

    # 전략 수익: T 종가에 신호 → T+1 보유 (shift(1))
    ew = close.pct_change(fill_method=None).clip(-0.3, 0.3).mean(axis=1)
    pos = pd.Series((state == 2).astype(float), index=d).shift(1)
    strat = ew * pos

    def idx_close(df: pd.DataFrame) -> pd.Series:
        col = [c for c in df.columns if "종가" in c or c == "Close"][0]
        return df[col].reindex(d).pct_change()
    bh_k = idx_close(kospi)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ── 성과 표 ────────────────────────────────────────────────
    rows = []
    for mask, period in [(d <= IS_END, "IS"), (d > IS_END, "OOS")]:
        m = mask.values
        for s, nm in [(strat, "강세만"), (ew, "EW보유"), (bh_k, "KOSPI")]:
            rows.append({**perf(s[m], nm), "period": period})
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "perf.csv", index=False)
    print(rep.to_string(index=False))

    # ── 국면별 개별주 수익 분포 ───────────────────────────────
    fwd_ret = close.pct_change(20, fill_method=None).shift(-20).clip(-0.5, 0.5).median(axis=1)
    dist_df = pd.DataFrame({"state": state, "fwd": fwd_ret.values,
                            "oos": (d > IS_END).values}).dropna()
    tbl = dist_df.groupby(["oos", "state"])["fwd"].agg(["count","mean","median"])
    print(tbl)
    tbl.to_csv(out / "regime_fwd_dist.csv")

    # ── 국면 판정 결과 저장 ───────────────────────────────────
    res.to_parquet(out / "regime_daily.parquet")

    # ── 시각화 ────────────────────────────────────────────────
    lab = {2: "강세", 1: "박스권", 0: "약세"}
    col = {2: NEON[2], 1: NEON[3], 0: NEON[1]}

    ki_col = [c for c in kospi.columns if "종가" in c or c == "Close"][0]
    ki = kospi[ki_col].reindex(d)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[.35, .15, .2, .3],
                        subplot_titles=["KOSPI + 국면", "50일선 위 종목비율",
                                        "분배일(25일 합산)", "누적수익"])

    for s in [2, 1, 0]:
        m = state == s
        fig.add_trace(go.Scatter(x=d[m], y=ki.values[m], mode="markers",
                                 marker=dict(size=3, color=col[s]), name=lab[s]), 1, 1)

    fig.add_trace(go.Scatter(x=d, y=res["p50"].values,
                             line=dict(color=NEON[0], width=1), name="p50"), 2, 1)
    fig.add_hline(y=cfg.b_hi, line_dash="dot", line_color=NEON[2], row=2, col=1)
    fig.add_hline(y=cfg.b_lo, line_dash="dot", line_color=NEON[1], row=2, col=1)

    fig.add_trace(go.Bar(x=d, y=res["dd_k"].values,
                         marker_color=NEON[4], name="분배일(KOSPI)"), 3, 1)

    for s, nm, c in [(strat, "강세만", NEON[2]), (ew, "EW보유", NEON[4]), (bh_k, "KOSPI", NEON[0])]:
        fig.add_trace(go.Scatter(x=d, y=(1 + s.fillna(0)).cumprod(),
                                 name=nm, line=dict(color=c)), 4, 1)

    fig.add_vline(x=pd.Timestamp(IS_END), line_dash="dash", line_color="#888")
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=4, col=1)
    _layout(fig, f"시장 국면 판정 (ma={cfg.ma_n}, b_hi={cfg.b_hi}, confirm={cfg.confirm})", 900)
    fig.write_html(out / "regime_chart.html")
    print(f"차트 저장: {out}/regime_chart.html")

if __name__ == "__main__":
    main()
