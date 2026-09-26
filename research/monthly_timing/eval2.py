import pickle, numpy as np, pandas as pd
from core import *
from strategies import freeze_rule
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
R=pickle.load(open('wf.pkl','rb')); r=R['r']; f=R['f']; m=R['m']; fam=R['fam']
WF0=('2019-01'); ONE=pd.Series(1.0,index=r.index)
def dd_rule(W,X):
    dd=m['kospi']/m['kospi'].rolling(W).max()-1
    return (dd>-X/100).astype(float).where(dd.notna())
# 1) perturbation grid on DEV (fixed params, 20bps) + WF-OOS
rows=[]
for W in [6,9,12,15,18]:
    for X in [2,3,4,5,6,7,8,10,15]:
        p=dd_rule(W,X)
        a=metrics(strat_returns(p,r,20).loc[EVAL_START:DEV_END],p)
        b=metrics(strat_returns(p,r,20).loc[WF0:DEV_END],p)
        rows.append((W,X,a['Sharpe'],b['Sharpe'],a['MDD'],a['Exposure']))
P=pd.DataFrame(rows,columns=['W','X','Sh_dev','Sh_wf','MDD_dev','Exp_dev'])
print(P.pivot(index='W',columns='X',values='Sh_dev').round(2)); print(P.pivot(index='W',columns='X',values='Sh_wf').round(2))
bh=metrics(r.loc[EVAL_START:DEV_END]); print('B&H dev Sharpe',round(bh['Sharpe'],3))
# MA perturbation
print('PxMA N dev/wf sharpe 20bps')
for N in [6,7,8,9,10,11,12,13,14,18,24]:
    p=fam['PxMA'][f'PxMA{N}']
    print(N, round(metrics(strat_returns(p,r,20).loc[EVAL_START:DEV_END])['Sharpe'],3), round(metrics(strat_returns(p,r,20).loc[WF0:DEV_END])['Sharpe'],3))
print('freeze DD family:',freeze_rule(fam['DD'],r),' freeze PxMA:',freeze_rule(fam['PxMA'],r),' freeze ALL:',freeze_rule({k:v for d in fam.values() for k,v in d.items()},r))
pickle.dump(P,open('pert.pkl','wb'))
