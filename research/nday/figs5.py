"""5차 시각화: 리밸런싱 주기별 누적수익(50bp) — holdout 음영."""
import pickle, plotly.graph_objects as go
res = pickle.load(open("run5.pkl", "rb"))
fig = go.Figure()
for n in ["M", 5, 10]:
    x = res[n][0].loc["2019-01":"2026-08"].fillna(0)
    fig.add_trace(go.Scatter(x=x.index.to_timestamp(), y=(1 + x).cumprod(), name=f"n={n}"))
fig.add_vrect(x0="2024-07-01", x1="2026-08-31", fillcolor="gray", opacity=0.15, annotation_text="holdout(진단)")
fig.update_layout(title="52WH 리밸런싱 주기별 누적수익 (50bp)", yaxis_type="log", template="plotly_dark")
fig.write_html("figs5.html", include_plotlyjs="cdn")
