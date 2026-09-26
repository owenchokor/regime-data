import pickle, numpy as np, pandas as pd
from exp2 import rmetrics, OBJECTIVES
from core import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300)
X=pickle.load(open('exp2.pkl','rb')); rf=X['rf']; r=X['r']; grid=X['grid']; ig=X['idxgrid']
WF=('2019-01',DEV_END); cols=['CAGR','Vol','Sharpe','Sortino','MDD','Calmar','CVaR5','MaxUW']
def T(d,a,b): return pd.DataFrame({k:rmetrics(v.loc[a:b],rf) for k,v in d.items()}).T[cols]
bh={'KOSPI B&H':r}
print('== (A) index timing WF-OOS, nested by objective (20bps, cash=CD)')
d=dict(bh); d.update({f'N[{o}]':X['A']['nested'][o][0] for o in OBJECTIVES}); d['DD12>-5% fixed']=ig[20]['DD12>-5%']
print(T(d,*WF).round(3))
for o in OBJECTIVES: print(o, X['A']['nested'][o][1], 'freeze', X['A']['freeze'][o])
print('== (B) stocks WF-OOS, nested by objective (50bps)')
d=dict(bh); d['UNIV-EW (top500 EW)']=grid[50]['UNIV-EW|ALWAYS']
d.update({f'N[{o}]':X['B']['nested'][o][0] for o in OBJECTIVES})
print(T(d,*WF).round(3))
print('== (B) grid WF-OOS top 25 by Sharpe (50bps) + ungated strategies')
G=T(grid[50],*WF).sort_values('Sharpe',ascending=False)
print(G.head(25).round(3)); print(G.loc[[k for k in G.index if k.endswith('|ALWAYS')]].round(3))
print('grid DEV (in-sample) top 10'); print(T(grid[50],EVAL_START,DEV_END).sort_values('Sharpe',ascending=False).head(10).round(3))
print('B&H dev', rmetrics(r.loc[EVAL_START:DEV_END],rf))
