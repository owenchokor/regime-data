import pickle, numpy as np, pandas as pd
from variants import *
from exp2 import rmetrics, rf, r
from core import *
X=pickle.load(open('exp2.pkl','rb')); gate=X['G']['PxMA10']; sim=Sim(); sector=load_sector()
WF=('2019-01',DEV_END); HO=(HOLD_START,LAST_MONTH); out={}
cov=sector.reindex(index=sim.S['M'].index).notna(); 
for cap in [4,3,6]:
    W=select_sector_cap(sim,cap,sector); mr,_=sim.month_returns(W,{},gate,rf)
    out[f'cap{cap}']=(mr-turnover_cost(W,gate).reindex(mr.index).fillna(0)).reindex(r.index)
base=pickle.load(open('run3.pkl','rb'))['res']['BASE'][0]; out['BASE']=base
for nm,(a,b) in [('WF',WF),('HOLD(진단)',HO)]:
    print(nm); print(pd.DataFrame({k:rmetrics(v.loc[a:b],rf) for k,v in out.items()}).T[['CAGR','Sharpe','Sortino','MDD','CVaR5']].round(3))
# 기준형의 업종 집중도
from exp2 import stock_weights
W=stock_weights()['52WH']; mx=[]
for t in W.index:
    tk=W.loc[t].dropna().index
    if len(tk) and t in sector.index: mx.append(sector.loc[t].reindex(tk).value_counts().max())
print('BASE 월별 최대 동일업종 종목수: median',np.median(mx),'p90',np.percentile(mx,90),'max',max(mx))
pickle.dump(out,open('run3b.pkl','wb'))
