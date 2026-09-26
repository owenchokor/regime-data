import pickle, numpy as np, pandas as pd
from core import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',200)
R=pickle.load(open('wf.pkl','rb')); r=R['r']; rq=R['rq']
WF0,WF1='2019-01',DEV_END
def tab(posd, s0, s1, cost=20, cols=('CAGR','Vol','Sharpe','MDD','Calmar','WinRate','Exposure','TurnoverYr')):
    rows={}
    for k,p in posd.items():
        x=strat_returns(p,r,cost).loc[s0:s1]
        rows[k]=metrics(x,p)
    return pd.DataFrame(rows).T[list(cols)]
base=dict(R['base_pos'])
print('== baseline full eval 2016-01~2026-08 (cost 0)')
t=tab(base,EVAL_START,LAST_MONTH,0,cols=('n','CAGR','Cum','Vol','Sharpe','MDD','Calmar','WinRate','MeanM','MedianM','DownDev','Worst','Best','Exposure','TurnoverYr'))
t.loc['KOSDAQ B&H']=pd.Series(metrics(rq.loc[EVAL_START:LAST_MONTH]))
print(t.round(3))
for nm,(a,b) in {'DEV':(EVAL_START,DEV_END),'WF-OOS':(WF0,WF1),'HOLD':(HOLD_START,LAST_MONTH)}.items():
    t=tab(base,a,b,0); t.loc['KOSDAQ B&H']=pd.Series(metrics(rq.loc[a:b])); print('==',nm,a,b); print(t.round(3))
cand={}
for (fam,w),(p,picks) in R['nested'].items(): cand[f'N:{fam}:{w[:3]}']=p
cand['Committee8']=R['committee']
for w,(o,dg) in R['ml'].items():
    for k,p in o.items(): cand[f'ML:{k}:{w[:3]}']=p
cand['B&H']=base['KOSPI B&H']
for c in [0,20,50]:
    t=tab(cand,WF0,WF1,c).sort_values('Sharpe',ascending=False)
    print(f'== WF-OOS {WF0}~{WF1} cost {c}bps  (n={len(r.loc[WF0:WF1])})'); print(t.round(3) if c==20 else t[['CAGR','Sharpe','MDD']].round(3).T)
pickle.dump(cand,open('cand.pkl','wb'))
print('== nested picks')
for (fam,w),(p,picks) in R['nested'].items(): print(fam,w,picks)
for w,(o,dg) in R['ml'].items():
    print(w,'feats',dg['feats'])
    for c in dg['coef']: print(w,c)
