"""4차 시각화: 기준형 vs 미국 게이트 누적수익 (WF+holdout, holdout 음영)."""
import pickle, pandas as pd, plotly.graph_objects as go
X = pickle.load(open("run4.pkl", "rb"))
fig = go.Figure()
for k, v in X["res"].items():
    x = v.loc["2019-01":"2026-08"].fillna(0)
    fig.add_trace(go.Scatter(x=x.index.to_timestamp(), y=(1 + x).cumprod(), name=k))
fig.add_vrect(x0="2024-07-01", x1="2026-08-31", fillcolor="gray", opacity=0.15, annotation_text="holdout(진단)")
fig.update_layout(title="52WH × 게이트 누적수익 (50bp)", yaxis_type="log", template="plotly_dark")
fig.write_html("figs4.html", include_plotlyjs="cdn")
