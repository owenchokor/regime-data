import pickle, time, numpy as np, pandas as pd
from stocks import *
t0=time.time(); c,s,v,mc=load_panels()
ra,E=adjusted_returns(c,s); print('adj done',round(time.time()-t0),'s', E.type.value_counts().to_dict())
print('matched lag dist', E[E.type=='matched'].lag.describe().round(1).to_dict())
um=E[E.type=='unmatched']; print('unmatched raw_ret quantiles', um.raw_ret.quantile([.01,.1,.5,.9,.99]).round(2).to_dict())
MP=monthly_panel(ra,c,v,mc,um)
print('eligible per month (first/mid/last)', MP['elig'].sum(1).iloc[[13,80,-2]].tolist())
f=MP['fwd']; e=MP['elig']
print('fwd abs>100% among eligible top500:', int(((f.abs()>1)&(MP['rank']<=500)).sum().sum()))
pickle.dump(dict(MP=MP,E=E),open('stocks.pkl','wb')); print('saved',round(time.time()-t0))
