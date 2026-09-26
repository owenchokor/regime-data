"""Plotly 다크 테마 차트 → HTML 조각."""
import pickle, numpy as np, pandas as pd, plotly.graph_objects as go
from core import *
BG="#070b14"; PAPER="#05080f"; GRID="#1a2233"; TEXT="#c9d4e5"
NEON=["#00e5ff","#ff2e88","#39ff14","#ffd400","#b388ff","#ff8a00","#7cf6ff"]
def _layout(fig,title,h=420,**kw):
    fig.update_layout(title=dict(text=title,font=dict(size=15)),template="plotly_dark",plot_bgcolor=BG,paper_bgcolor=PAPER,
        font=dict(color=TEXT,family="Pretendard, Noto Sans KR, sans-serif",size=12),height=h,margin=dict(l=50,r=20,t=50,b=40),
        legend=dict(orientation="h",y=-0.18,bgcolor="rgba(0,0,0,0)"),**kw)
    fig.update_xaxes(gridcolor=GRID,zeroline=False); fig.update_yaxes(gridcolor=GRID,zeroline=False); return fig
R=pickle.load(open('wf.pkl','rb')); r=R['r']; fam=R['fam']; f=R['f']; T=pickle.load(open('tests.pkl','rb'))
cand=pickle.load(open('cand.pkl','rb')); P=pickle.load(open('pert.pkl','rb')); E4=pickle.load(open('eval4.pkl','rb'))
FINAL=fam['DD']['DD12>-5%']; ONE=pd.Series(1.0,index=r.index)
idx=r.loc[EVAL_START:LAST_MONTH].index; ts=idx.to_timestamp(how='end').normalize()
figs={}
def shade(fig):
    fig.add_vrect(x0=pd.Timestamp('2019-01-01'),x1=pd.Timestamp('2024-06-30'),fillcolor="#00e5ff",opacity=0.06,line_width=0,annotation_text="Walk-forward OOS",annotation_position="top left",annotation_font_color="#7cf6ff")
    fig.add_vrect(x0=pd.Timestamp('2024-07-01'),x1=ts[-1],fillcolor="#ff2e88",opacity=0.08,line_width=0,annotation_text="Holdout",annotation_position="top left",annotation_font_color="#ff2e88")
S={'KOSPI B&H':ONE,'DD12>-5% (최종)':FINAL,'PxMA10':fam['PxMA']['PxMA10'],'TSMOM12':fam['TSMOM']['TSMOM12']}
eq={k:(1+strat_returns(p,r,20).loc[idx].fillna(0)).cumprod() for k,p in S.items()}
fig=go.Figure()
for i,(k,e) in enumerate(eq.items()):
    fig.add_trace(go.Scatter(x=ts,y=e.values,name=k,line=dict(color=NEON[i],width=3 if i<2 else 1.5,dash=None if i<2 else 'dot')))
pos=FINAL.shift(1).loc[idx]
fig.add_trace(go.Scatter(x=ts,y=pos.values,name='최종 룰 포지션',yaxis='y2',line=dict(color='#39ff14',width=0),fill='tozeroy',fillcolor='rgba(57,255,20,0.12)',line_shape='hv'))
shade(fig); _layout(fig,"누적 성과 (20bps 비용, 로그축) — 초록 음영 = 최종 룰 투자 월",h=460,yaxis=dict(type='log',title='누적(1=시작)'),yaxis2=dict(overlaying='y',side='right',range=[0,4],showticklabels=False,showgrid=False))
figs['equity']=fig
fig=go.Figure()
for i,(k,e) in enumerate(eq.items()):
    fig.add_trace(go.Scatter(x=ts,y=(e/e.cummax().clip(lower=1)-1).values*100,name=k,line=dict(color=NEON[i],width=2 if i<2 else 1,dash=None if i<2 else 'dot')))
shade(fig); _layout(fig,"고점 대비 낙폭(%)",yaxis=dict(title='%')); figs['dd']=fig
# 후보 막대
a,b='2019-01',DEV_END
rows={k:metrics(strat_returns(p,r,20).loc[a:b],p) for k,p in cand.items()}
D=pd.DataFrame(rows).T.sort_values('Sharpe')
col=['#ffd400' if k=='B&H' else ('#39ff14' if k=='N:DD:exp' else ('#b388ff' if k.startswith('ML') else '#00e5ff')) for k in D.index]
fig=go.Figure(go.Bar(x=D['Sharpe'],y=D.index,orientation='h',marker_color=col,customdata=np.c_[D['CAGR']*100,D['MDD']*100,D['Exposure']],hovertemplate='%{y}<br>Sharpe %{x:.2f}<br>CAGR %{customdata[0]:.1f}%<br>MDD %{customdata[1]:.1f}%<br>노출 %{customdata[2]:.2f}<extra></extra>'))
fig.add_vline(x=D.loc['B&H','Sharpe'],line_color='#ffd400',line_dash='dash',annotation_text='B&H',annotation_font_color='#ffd400')
_layout(fig,"Walk-forward OOS(2019-01~2024-06) Sharpe — 71개 설정 (보라=ML, 파랑=룰, 초록=최종)",h=1100,yaxis=dict(tickfont=dict(size=9)),showlegend=False); figs['cands']=fig
# 섭동 히트맵
Z=P.pivot(index='W',columns='X',values='Sh_dev')
fig=go.Figure(go.Heatmap(z=Z.values,x=[f'-{c}%' for c in Z.columns],y=[f'{w}M' for w in Z.index],colorscale=[[0,'#1a0033'],[0.5,'#2a3a5a'],[1,'#39ff14']],zmid=metrics(r.loc[EVAL_START:DEV_END])['Sharpe'],text=np.round(Z.values,2),texttemplate='%{text}',colorbar=dict(title='Sharpe')))
_layout(fig,"파라미터 섭동: 고점대비 낙폭 룰 (개발구간 Sharpe, B&H=0.33)",h=380,xaxis=dict(title='허용 낙폭 X'),yaxis=dict(title='고점 창 W')); figs['pert']=fig
# 순열 null
t=T['WF+HOLD']
fig=go.Figure(go.Histogram(x=t['perm'],nbinsx=25,marker_color='#b388ff',opacity=0.8,name='무작위 타이밍(원형 이동)'))
fig.add_vline(x=t['sh'],line_color='#39ff14',line_width=3,annotation_text=f"최종 룰 {t['sh']:.2f}",annotation_font_color='#39ff14')
fig.add_vline(x=t['bh'],line_color='#ffd400',line_dash='dash',annotation_text=f"B&H {t['bh']:.2f}",annotation_position='bottom right',annotation_font_color='#ffd400')
_layout(fig,f"순열검정: 같은 포지션 패턴을 시간축으로 밀었을 때 Sharpe 분포 (p={t['p_perm']:.2f})",h=360,xaxis=dict(title='Sharpe'),showlegend=False); figs['perm']=fig
# 비용 민감도
ft=pickle.load(open('final_tables.pkl','rb'))
fig=go.Figure()
for i,per in enumerate(['WF-OOS','HOLDOUT']):
    tt=ft[per]; fig.add_trace(go.Bar(x=[f'{c}bp' for c in COSTS_BPS],y=[tt.loc[f'DD_{c}bp','Sharpe'] for c in COSTS_BPS],name=f'최종 룰 · {per}',marker_color=NEON[i]))
    fig.add_hline(y=tt.loc['B&H','Sharpe'],line_color=NEON[i],line_dash='dot',annotation_text=f'B&H {per}',annotation_font_color=NEON[i])
_layout(fig,"거래비용별 Sharpe (점선=같은 구간 B&H)",h=360,barmode='group'); figs['cost']=fig
# 연도별
yrs=list(range(2016,2027)); xs=strat_returns(FINAL,r,20)
dd_y=[(1+xs.loc[str(y)].dropna()).prod()-1 for y in yrs]; bh_y=[(1+r.loc[str(y)]).prod()-1 for y in yrs]
fig=go.Figure([go.Bar(x=yrs,y=np.array(bh_y)*100,name='B&H',marker_color='#ffd400'),go.Bar(x=yrs,y=np.array(dd_y)*100,name='최종 룰',marker_color='#39ff14')])
_layout(fig,"연도별 수익률(%) — 2026은 8월까지",h=360,barmode='group'); figs['yearly']=fig
# 상관
C=E4['C']; fig=go.Figure(go.Heatmap(z=C.values,x=C.columns,y=C.index,zmin=-1,zmax=1,colorscale=[[0,'#ff2e88'],[0.5,BG],[1,'#00e5ff']],text=np.round(C.values,2),texttemplate='%{text}',textfont=dict(size=9)))
_layout(fig,"ML 후보 피처 상관 (2015-12~2024-06)",h=460); figs['corr']=fig
html={k:v.to_html(full_html=False,include_plotlyjs=False,config={'displayModeBar':False,'responsive':True}) for k,v in figs.items()}
pickle.dump(html,open('figs_html.pkl','wb')); print({k:len(v) for k,v in html.items()})
