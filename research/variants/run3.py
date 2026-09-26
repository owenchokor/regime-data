import pickle, numpy as np, pandas as pd, time
from variants import *
from exp2 import rmetrics, rf, r
from core import *
X=pickle.load(open('exp2.pkl','rb')); G=X['G']; gate=G['PxMA10']
sim=Sim(); WF=('2019-01',DEV_END); HO=(HOLD_START,LAST_MONTH)
def run(opt):
    W=sim.select(opt); mr,st=sim.month_returns(W,opt,gate,rf)
    x=(mr - turnover_cost(W,gate).reindex(mr.index).fillna(0)).reindex(r.index)
    return x,st
res={}; t0=time.time()
for k,opt in VARIANTS.items():
    res[k]=run(opt); print(k,round(time.time()-t0))
nb={}
for k,lst in NEIGHBORS.items():
    for opt in lst: nb[(k,str(opt))]=run(opt)[0]
pickle.dump(dict(res=res,nb=nb),open('run3.pkl','wb'))
cols=['CAGR','Vol','Sharpe','Sortino','MDD','CVaR5','MaxUW']
for nm,(a,b) in [('WF',WF),('HOLD(진단)',HO)]:
    T=pd.DataFrame({k:rmetrics(v[0].loc[a:b],rf) for k,v in res.items()}).T[cols]; T['stop_frac']=[v[1].reindex(r.loc[a:b].index).mean() for v in res.values()]
    print('==',nm); print(T.round(3))
print('base prev grid WF Sharpe', round(rmetrics(X['grid'][50]['52WH|PxMA10'].loc[WF[0]:WF[1]],rf)['Sharpe'],3))
print('== neighbors WF'); print(pd.DataFrame({str(k):rmetrics(v.loc[WF[0]:WF[1]],rf) for k,v in nb.items()}).T[['Sharpe','MDD','CVaR5']].round(3))
