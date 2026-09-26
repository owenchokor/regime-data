"""8차 시각화: A(채점 후 익일 매수) vs B(20개 선매수 후 거르기) — IC별 KOSPI 대비 초과수익(k=5, 전 기간 중앙값)."""
import pandas as pd, plotly.graph_objects as go
D = pd.read_pickle("run8.pkl").rename(columns={"mode": "md"}); D = D[(D.per == "ALL") & (D.k == 5)]
fig = go.Figure()
for m in ["A", "B"]:
    g = D[D.md == m].groupby("ic").ex.median()
    fig.add_trace(go.Scatter(x=g.index, y=g.values * 100, name=m))
fig.add_hline(y=0); fig.update_layout(title="체결 현실화 스펙트럼 (k=5)", xaxis_title="IC", yaxis_title="연 초과수익 %p", template="plotly_dark")
fig.write_html("figs8.html", include_plotlyjs="cdn")
