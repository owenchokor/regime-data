import pickle, numpy as np, pandas as pd
from core import *
R=pickle.load(open('wf.pkl','rb')); r=R['r']; rq=R['rq']; fam=R['fam']; H=pickle.load(open('figs_html.pkl','rb'))
T=pickle.load(open('tests.pkl','rb')); ft=pickle.load(open('final_tables.pkl','rb'))
ONE=pd.Series(1.0,index=r.index); FINAL=fam['DD']['DD12>-5%']
pct=lambda v: f"{v*100:.1f}%"; f2=lambda v: f"{v:.2f}"
FMT={'CAGR':pct,'Cum':pct,'Vol':pct,'Sharpe':f2,'MDD':pct,'Calmar':f2,'WinRate':pct,'MeanM':pct,'MedianM':pct,'DownDev':pct,'Worst':pct,'Best':pct,'Exposure':f2,'TurnoverYr':f2,'n':lambda v:f"{int(v)}"}
KO={'n':'월수','CAGR':'연복리','Cum':'누적','Vol':'연변동성','Sharpe':'Sharpe','MDD':'MDD','Calmar':'Calmar','WinRate':'월승률','MeanM':'월평균','MedianM':'월중앙값','DownDev':'하방편차','Worst':'최악월','Best':'최고월','Exposure':'평균노출','TurnoverYr':'연회전'}
def tbl(df,cols):
    df=df[cols].copy()
    for c in cols: df[c]=df[c].map(lambda v: '–' if pd.isna(v) else FMT[c](v))
    df.columns=[KO[c] for c in cols]; return df.to_html(classes='t',border=0)
base={'KOSPI B&H':ONE,'TSMOM1 (월간 모멘텀)':fam['TSMOM']['TSMOM1'],'TSMOM12':fam['TSMOM']['TSMOM12'],'PxMA10 (이평 타이밍)':fam['PxMA']['PxMA10']}
allc=['n','CAGR','Cum','Vol','Sharpe','MDD','Calmar','WinRate','MeanM','MedianM','DownDev','Worst','Best','Exposure','TurnoverYr']
def btab(a,b,cost):
    d={k:metrics(strat_returns(p,r,cost).loc[a:b],p) for k,p in base.items()}; d['KOSDAQ B&H']=metrics(rq.loc[a:b]); return pd.DataFrame(d).T
B_full=tbl(btab(EVAL_START,LAST_MONTH,0),allc)
B_dev=tbl(btab(EVAL_START,DEV_END,0),['CAGR','Vol','Sharpe','MDD','Calmar','Exposure'])
B_hold=tbl(btab(HOLD_START,LAST_MONTH,0),['CAGR','Vol','Sharpe','MDD','Calmar','Exposure'])
# 폴드 표
xs=strat_returns(FINAL,r,20); rows=[]
pk_all=dict((y,n) for y,n,_ in R['nested'][('ALL','expanding')][1])
for y,n,s in R['nested'][('DD','expanding')][1]:
    o1=DEV_END if y==2024 else f'{y}-12'
    rows.append([f'{y}'+(' (1~6월)' if y==2024 else ''),n,f'{s:.2f}',pct((1+xs.loc[f'{y}-01':o1]).prod()-1),pct((1+r.loc[f'{y}-01':o1]).prod()-1),pk_all[y]])
FOLD=pd.DataFrame(rows,columns=['OOS 연도','DD 패밀리 선택','훈련 Sharpe','OOS 수익(룰)','OOS 수익(B&H)','전체 룰 중 선택(참고)']).to_html(index=False,classes='t',border=0)
fc=['n','CAGR','Cum','Vol','Sharpe','MDD','Calmar','WinRate','DownDev','Worst','Exposure','TurnoverYr']
FT={k:tbl(v,fc) for k,v in ft.items()}
def trow(k,t): return f"<tr><td>{k}</td><td>{t['n']}</td><td>{t['sh']:.2f} [{t['sh_lo']:.2f}, {t['sh_hi']:.2f}]</td><td>{t['bh']:.2f}</td><td>{t['dsh']:+.2f} [{t['dsh_lo']:+.2f}, {t['dsh_hi']:+.2f}]</td><td>{t['p_boot']:.2f}</td><td>{t['p_perm']:.2f}</td><td>{t['diff']*100:+.2f}%p (t={t['nw_t']:.2f}, p={t['nw_p']:.2f})</td><td>{t['alpha']*100:+.2f}%p (t={t['alpha_t']:.2f}), β={t['beta']:.2f}</td></tr>"
ST="<table class='t'><tr><th>구간</th><th>월수</th><th>Sharpe [95% CI]</th><th>B&H</th><th>ΔSharpe [95% CI]</th><th>p(부트)</th><th>p(순열)</th><th>월 초과수익 (NW)</th><th>타이밍 알파·베타</th></tr>"+''.join(trow(k,t) for k,t in T.items())+"</table>"
tpl=open('report_tpl.html').read()
for k,v in dict(B_full=B_full,B_dev=B_dev,B_hold=B_hold,FOLD=FOLD,FT_WF=FT['WF-OOS'],FT_HOLD=FT['HOLDOUT'],FT_FULL=FT['FULL'],STATS=ST).items(): tpl=tpl.replace('{{'+k+'}}',v)
for k,v in H.items(): tpl=tpl.replace('{{F_'+k+'}}',f'<div class="fig">{v}</div>')
open('monthly_timing_report.html','w').write(tpl); print('ok', len(tpl))
