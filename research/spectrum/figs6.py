"""6차 시각화: IC × k 별 초과수익(중앙값), 후보풀 52WH vs 무작위."""
import pandas as pd, plotly.graph_objects as go
D = pd.read_pickle("run6.pkl"); D = D[D.per == "ALL"]
fig = go.Figure()
for pool, dash in [("52WH", "solid"), ("RAND", "dot")]:
    for k in [3, 5, 10]:
        g = D[(D.pool == pool) & (D.k == k) & (D.ic.between(-0.3, 0.5))].groupby("ic").exKOSPI.median()
        fig.add_trace(go.Scatter(x=g.index, y=g.values * 100, name=f"{pool} k={k}", line=dict(dash=dash)))
fig.add_hline(y=0); fig.update_layout(title="재선별 스펙트럼: 선택 IC별 연 초과수익(%p, 2016–2026-08 중앙값)", xaxis_title="IC", template="plotly_dark")
fig.write_html("figs6.html", include_plotlyjs="cdn")
