import pickle, numpy as np, pandas as pd, plotly.graph_objects as go
from plotly.subplots import make_subplots
from figs import _layout, BG, NEON, shade
from exp2 import rmetrics, rf, r
from variants import load_sector
from exp2 import stock_weights
from core import *
WF=('2019-01',DEV_END); HO=(HOLD_START,LAST_MONTH)
D=pickle.load(open('run3.pkl','rb')); B=pickle.load(open('run3b.pkl','rb'))
V={k:v[0] for k,v in D['res'].items()}; V['V7 업종 상한(업종당 최대 4)']=B['cap4']
try:
    V8=pickle.load(open('run3c.pkl','rb')); V.update(V8['main'])
except FileNotFoundError: V8=None
figs={}
# 1) 판정 산점도: WF Sharpe vs WF MDD
base=rmetrics(V['BASE'].loc[WF[0]:WF[1]],rf)
fig=go.Figure()
for i,(k,x) in enumerate(V.items()):
    m=rmetrics(x.loc[WF[0]:WF[1]],rf); h=rmetrics(x.loc[HO[0]:HO[1]],rf)
    fig.add_trace(go.Scatter(x=[m['MDD']*100],y=[m['Sharpe']],mode='markers+text',text=[k.split(' ')[0]],textposition='top center',marker=dict(size=14,color=NEON[i%7],line=dict(width=1,color='#fff')),name=k,
        hovertemplate=f"{k}<br>WF Sharpe %{{y:.2f}}<br>WF MDD %{{x:.1f}}%<br>CVaR5 {m['CVaR5']*100:.1f}%<br>(holdout 진단 Sharpe {h['Sharpe']:.2f})<extra></extra>"))
fig.add_hline(y=base['Sharpe']-0.05,line_dash='dash',line_color='#ffd400',annotation_text='통과선: 기준형 Sharpe −0.05',annotation_font_color='#ffd400')
fig.add_vline(x=(base['MDD']+0.03)*100,line_dash='dot',line_color='#ff2e88',annotation_text='MDD 3%p 개선선',annotation_font_color='#ff2e88',annotation_position='bottom right')
_layout(fig,"사전등록 변형 판정 (WF-OOS 2019-01~2024-06, 50bp) — 우상단 사분면이 통과",h=460,xaxis=dict(title='MDD (%)'),yaxis=dict(title='Sharpe (초과)')); figs['judge']=fig
# 2) WF vs holdout 진단 막대
ks=list(V); wf=[rmetrics(V[k].loc[WF[0]:WF[1]],rf)['Sharpe'] for k in ks]; ho=[rmetrics(V[k].loc[HO[0]:HO[1]],rf)['Sharpe'] for k in ks]
fig=go.Figure([go.Bar(x=ks,y=wf,name='WF-OOS (판정용)',marker_color='#00e5ff'),go.Bar(x=ks,y=ho,name='Holdout (오염된 진단용)',marker_color='#ff2e88',opacity=.6)])
_layout(fig,"WF에서 나쁜 변형이 holdout에서 좋아 보이는 역전 — 이미 본 급락에 맞춘 효과",h=420,barmode='group',xaxis=dict(tickangle=-25)); figs['flip']=fig
# 3) 스톱 폭 곡선
nb=D['nb']; xs=[.10,.15,.20]; ser={.10:nb[('V4 종목 트레일링 스톱 15%',"{'stock_stop': 0.1}")],.15:V['V4 종목 트레일링 스톱 15%'],.20:nb[('V4 종목 트레일링 스톱 15%',"{'stock_stop': 0.2}")]}
fig=make_subplots(rows=1,cols=2,subplot_titles=('Sharpe','MDD (%)'))
for per,col,(a,b) in [('WF','#00e5ff',WF),('Holdout(진단)','#ff2e88',HO)]:
    fig.add_trace(go.Scatter(x=[f'{int(x*100)}%' for x in xs],y=[rmetrics(ser[x].loc[a:b],rf)['Sharpe'] for x in xs],name=per,line=dict(color=col,width=3),mode='lines+markers'),1,1)
    fig.add_trace(go.Scatter(x=[f'{int(x*100)}%' for x in xs],y=[rmetrics(ser[x].loc[a:b],rf)['MDD']*100 for x in xs],name=per,showlegend=False,line=dict(color=col,width=3),mode='lines+markers'),1,2)
_layout(fig,"종목 트레일링 스톱 폭별 성과 (기준형 WF Sharpe 0.91, MDD −18.1%)",h=360); figs['stop']=fig
# 4) 업종 집중
sector=load_sector(); W=stock_weights()['52WH']; rows=[]
for t in W.loc['2016-01':].index:
    w=W.loc[t]; tk=w[w>0].index
    if len(tk) and t in sector.index:
        vc=sector.loc[t].reindex(tk).value_counts(); rows.append((t.to_timestamp(how='end').normalize(),vc.max(),vc.index[0]))
Z=pd.DataFrame(rows,columns=['t','mx','sec'])
fig=go.Figure(go.Bar(x=Z.t,y=Z.mx,marker_color=np.where(Z.mx>=8,'#ff2e88','#00e5ff'),customdata=Z.sec,hovertemplate='%{x|%Y-%m}<br>%{customdata}: %{y}종목<extra></extra>'))
shade(fig); _layout(fig,"기준형 20종목 중 최대 동일업종 종목수 (빨강 ≥8)",h=340); figs['sector']=fig
html={k:v.to_html(full_html=False,include_plotlyjs=False,config={'displayModeBar':False,'responsive':True}) for k,v in figs.items()}
pickle.dump(html,open('figs3_html.pkl','wb')); print({k:len(v) for k,v in html.items()})
