import pickle, numpy as np, pandas as pd
from exp2 import rmetrics, OBJECTIVES, rf, r
from core import *
X=pickle.load(open('exp2.pkl','rb')); grid=X['grid']; ig=X['idxgrid']; ST=pickle.load(open('ev2_tests.pkl','rb')); H=pickle.load(open('figs2_html.pkl','rb'))
WF=('2019-01',DEV_END); HO=(HOLD_START,LAST_MONTH)
pct=lambda v: f"{v*100:.1f}%"; f2=lambda v: f"{v:.2f}"
FM={'n':lambda v:f"{int(v)}",'CAGR':pct,'Vol':pct,'Sharpe':f2,'Sortino':f2,'MDD':pct,'Calmar':f2,'CVaR5':pct,'MaxUW':lambda v:f"{int(v)}",'WinRate':pct}
KO={'n':'월수','CAGR':'연복리','Vol':'연변동성','Sharpe':'Sharpe','Sortino':'Sortino','MDD':'MDD','Calmar':'Calmar','CVaR5':'CVaR 5%','MaxUW':'최장 수면하(월)','WinRate':'월승률'}
def tbl(d,a,b,cols=('CAGR','Vol','Sharpe','Sortino','MDD','Calmar','CVaR5','MaxUW')):
    df=pd.DataFrame({k:rmetrics(v.loc[a:b],rf) for k,v in d.items()}).T[list(cols)]
    for c in cols: df[c]=df[c].map(lambda v:'–' if pd.isna(v) else FM[c](v))
    df.columns=[KO[c] for c in cols]; return df.to_html(classes='t',border=0)
def picks(part):
    rows=[[o]+[p for _,p in X[part]['nested'][o][1]]+[X[part]['freeze'][o]] for o in OBJECTIVES]
    return pd.DataFrame(rows,columns=['목적함수']+[str(y) for y in WF_YEARS]+['동결(holdout용)']).to_html(index=False,classes='t',border=0)
dA={'KOSPI B&H':r}; dA.update({f'nested[{o}]':X['A']['nested'][o][0] for o in OBJECTIVES})
dB={'KOSPI B&H':r,'시총상위500 동일가중':grid[50]['UNIV-EW|ALWAYS'],'52WH 게이트없음':grid[50]['52WH|ALWAYS']}; dB.update({f'nested[{o}]':X['B']['nested'][o][0] for o in OBJECTIVES})
dH={'KOSPI B&H':r,'B 최종: 52WH × PxMA10 (50bp)':grid[50]['52WH|PxMA10'],'A 최종: TermUp 타이밍 (20bp)':ig[20]['TermUp'],'참고: 시총상위500 동일가중':grid[50]['UNIV-EW|ALWAYS'],'참고: 52WH × DD12>-5%':grid[50]['52WH|DD12>-5%'],'참고: 52WH 게이트없음':grid[50]['52WH|ALWAYS']}
def srow(k):
    t=ST[k]; return f"<tr><td>{k}</td><td>{t['n']}</td><td>{t['sh']:.2f}</td><td>{t['bh']:.2f}</td><td>{t['dsh']:+.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}]</td><td>{t['p']:.2f}</td><td>{t['alpha_ann']*100:+.1f}% (t={t['alpha_t']:.2f}, p={t['alpha_p']:.3f})</td><td>{t['beta']:.2f}</td></tr>"
STT="<table class='t'><tr><th>검정</th><th>월수</th><th>Sharpe</th><th>비교대상</th><th>ΔSharpe [95%]</th><th>p(부트)</th><th>연 알파 (NW)</th><th>β</th></tr>"+''.join(srow(k) for k in ['B WF (nested)','B HOLD','B WF+HOLD','52WH ungated WF','52WH ungated vs UNIV-EW WF','A WF (nested)','A HOLD'])+"</table>"
rows=[]
for c in [0,10,20,50]:
    a=rmetrics(grid[c]['52WH|PxMA10'].loc[HO[0]:HO[1]],rf); w=rmetrics(X['B']['nested']['CAGR|MDD'][0].loc[WF[0]:WF[1]]*0+grid[c]['52WH|PxMA10'].loc[WF[0]:WF[1]],rf)
    rows.append([f'{c}bp',f"{rmetrics(grid[c]['52WH|DD12>-5%'].loc[WF[0]:WF[1]],rf)['Sharpe']:.2f}",f"{w['Sharpe']:.2f}",f"{a['Sharpe']:.2f}",f"{a['CAGR']*100:.1f}%"])
COST=pd.DataFrame(rows,columns=['편도비용','WF Sharpe (52WH×DD)','WF Sharpe (52WH×PxMA10)','Holdout Sharpe','Holdout CAGR']).to_html(index=False,classes='t',border=0)
tpl=open('report2_tpl.html').read()
for k,v in dict(T_A=tbl(dA,*WF),T_B=tbl(dB,*WF),P_A=picks('A'),P_B=picks('B'),T_H=tbl(dH,*HO),STATS=STT,COST=COST).items(): tpl=tpl.replace('{{'+k+'}}',v)
for k,v in H.items(): tpl=tpl.replace('{{F_'+k+'}}',f'<div class="fig">{v}</div>')
open('gated_stock_risk_report.html','w').write(tpl); print('ok',len(tpl), tpl.count('{{'))
