import pickle, numpy as np, pandas as pd, plotly.graph_objects as go
from plotly.subplots import make_subplots
from figs import _layout, BG, NEON, shade
from exp2 import rmetrics, OBJECTIVES, rf, r, rate, term
from core import *
X=pickle.load(open('exp2.pkl','rb')); grid=X['grid']; ig=X['idxgrid']; ST=pickle.load(open('ev2_tests.pkl','rb')); C=pickle.load(open('ev2c.pkl','rb'))
WF=('2019-01',DEV_END); HO=(HOLD_START,LAST_MONTH)
idx=r.loc[EVAL_START:LAST_MONTH].index; ts=idx.to_timestamp(how='end').normalize()
Bpath=pd.concat([grid[50]['52WH|DD12>-5%'].loc[EVAL_START:'2018-12'].rename(None)*np.nan, X['B']['nested']['CAGR|MDD'][0].loc[WF[0]:WF[1]], grid[50]['52WH|PxMA10'].loc[HO[0]:HO[1]]])
Apath=pd.concat([X['A']['nested']['Sortino'][0].loc[WF[0]:WF[1]], ig[20]['TermUp'].loc[HO[0]:HO[1]]])
figs={}
S={'KOSPI B&H':r,'시총상위500 동일가중':grid[50]['UNIV-EW|ALWAYS'],'52WH 게이트없음':grid[50]['52WH|ALWAYS'],'52WH × KOSPI 게이트 (최종 절차)':Bpath,'KOSPI 타이밍 · TermUp (A 최종)':Apath}
fig=go.Figure()
for i,(k,x) in enumerate(S.items()):
    x=x.loc['2019-01':LAST_MONTH]; e=(1+x.fillna(0)).cumprod(); t=x.index.to_timestamp(how='end').normalize()
    fig.add_trace(go.Scatter(x=t,y=e.values,name=k,line=dict(color=NEON[[3,6,1,2,0][i]],width=3 if i in (0,3) else 1.6,dash=None if i in (0,3) else 'dot')))
shade(fig); _layout(fig,"누적 성과 2019-01~ (비용: 개별주 50bp·지수 20bp, 현금=CD91)",h=460,yaxis=dict(type='log')); figs['eq']=fig
fig=go.Figure()
for i,(k,x) in enumerate(S.items()):
    x=x.loc['2019-01':LAST_MONTH]; e=(1+x.fillna(0)).cumprod(); t=x.index.to_timestamp(how='end').normalize()
    fig.add_trace(go.Scatter(x=t,y=(e/e.cummax().clip(lower=1)-1).values*100,name=k,line=dict(color=NEON[[3,6,1,2,0][i]],width=2.5 if i in (0,3) else 1.2)))
shade(fig); _layout(fig,"고점 대비 낙폭(%)"); figs['dd']=fig
# 히트맵 전략×게이트
M=pd.Series({k:rmetrics(v.loc[WF[0]:WF[1]],rf)['Sharpe'] for k,v in grid[50].items()})
strs=list(dict.fromkeys(k.split('|')[0] for k in M.index)); gts=list(dict.fromkeys(k.split('|')[1] for k in M.index))
Z=np.array([[M[f'{s}|{g}'] for g in gts] for s in strs])
fig=go.Figure(go.Heatmap(z=Z,x=gts,y=strs,zmid=rmetrics(r.loc[WF[0]:WF[1]],rf)['Sharpe'],colorscale=[[0,'#ff2e88'],[0.5,BG],[1,'#39ff14']],text=np.round(Z,2),texttemplate='%{text}',textfont=dict(size=9),colorbar=dict(title='Sharpe')))
_layout(fig,"Walk-forward OOS Sharpe(초과·50bp): 종목전략 × 게이트 (중앙색=KOSPI 0.28)",h=520,xaxis=dict(tickangle=-40)); figs['heat']=fig
# 목적함수 비교
fig=make_subplots(rows=1,cols=2,subplot_titles=("Sharpe (초과수익)","MDD"))
for j,(lab,part,bench) in enumerate([('A 지수 타이밍','A',r),('B 개별주×게이트','B',r)]):
    sh=[rmetrics(X[part]['nested'][o][0].loc[WF[0]:WF[1]],rf)['Sharpe'] for o in OBJECTIVES]; md=[rmetrics(X[part]['nested'][o][0].loc[WF[0]:WF[1]],rf)['MDD']*100 for o in OBJECTIVES]
    fig.add_trace(go.Bar(x=OBJECTIVES,y=sh,name=lab,marker_color=NEON[j]),1,1); fig.add_trace(go.Bar(x=OBJECTIVES,y=md,name=lab,marker_color=NEON[j],showlegend=False),1,2)
bm=rmetrics(r.loc[WF[0]:WF[1]],rf); fig.add_hline(y=bm['Sharpe'],line_dash='dot',line_color='#ffd400',row=1,col=1); fig.add_hline(y=bm['MDD']*100,line_dash='dot',line_color='#ffd400',row=1,col=2)
_layout(fig,"Nested 선택 목적함수별 walk-forward OOS 성과 (점선 = KOSPI B&H)",h=400,barmode='group'); figs['obj']=fig
# 랜덤 포트폴리오
fig=make_subplots(rows=1,cols=2,subplot_titles=("Walk-forward OOS","Holdout"))
for j,per in enumerate(['WF','HOLD']):
    obs,dist=ST[f'rand_PxMA10_{per}']
    fig.add_trace(go.Histogram(x=dist,nbinsx=25,marker_color='#b388ff',opacity=.8,showlegend=False),1,j+1)
    fig.add_vline(x=obs,line_color='#39ff14',line_width=3,row=1,col=j+1,annotation_text=f'52WH {obs:.2f}',annotation_font_color='#39ff14')
_layout(fig,"같은 유니버스·같은 게이트에서 무작위 20종목 300회 Sharpe 분포 vs 52WH (비용 0)",h=360); figs['rand']=fig
# 강건성 히트맵
R=C['R']; R=R[R.gate=='PxMA10']
fig=make_subplots(rows=1,cols=2,subplot_titles=("WF-OOS Sharpe","Holdout Sharpe (사후 진단)"))
for j,col in enumerate(['WF_Sh','HO_Sh']):
    P=R.pivot(index='univ',columns='topn',values=col)
    fig.add_trace(go.Heatmap(z=P.values,x=[f'N={c}' for c in P.columns],y=[f'상위{u}' for u in P.index],zmid=0.28 if j==0 else 1.09,colorscale=[[0,'#ff2e88'],[0.5,BG],[1,'#39ff14']],text=np.round(P.values,2),texttemplate='%{text}',showscale=False),1,j+1)
_layout(fig,"52WH × PxMA10: 유니버스·보유종목수 섭동 (50bp)",h=340); figs['rob']=fig
yr=C['yr']; Y=yr.groupby(yr.index.year).apply(lambda d:(1+d).prod()-1)
fig=go.Figure([go.Bar(x=Y.index,y=Y[c]*100,name=c,marker_color=NEON[[2,1,3,6][i]]) for i,c in enumerate(Y.columns)])
_layout(fig,"연도별 수익률(%) — 2026은 8월까지",h=360,barmode='group'); figs['yr']=fig
# ECOS
t2=rate.index.to_timestamp(how='end').normalize(); sel=rate.index>=pd.Period('2014-01','M')
fig=make_subplots(specs=[[{"secondary_y":True}]])
for i,c in enumerate(['CD(91일)','국고채(3년)','국고채(10년)']): fig.add_trace(go.Scatter(x=t2[sel],y=rate[c][sel],name=c,line=dict(color=NEON[i],width=1.8)),secondary_y=False)
fig.add_trace(go.Scatter(x=t2[sel],y=(rate['국고채(10년)']-rate['국고채(3년)'])[sel],name='장단기 스프레드(10Y−3Y)',line=dict(color='#ffd400',width=1.2,dash='dot')),secondary_y=True)
_layout(fig,"ECOS 시장금리(월평균, %) — CD91을 현금수익·무위험수익으로 사용",h=380); figs['ecos']=fig
html={k:v.to_html(full_html=False,include_plotlyjs=False,config={'displayModeBar':False,'responsive':True}) for k,v in figs.items()}
pickle.dump(html,open('figs2_html.pkl','wb')); print({k:len(v) for k,v in html.items()})
