import pickle, numpy as np, pandas as pd
import exp2
from exp2 import rmetrics, S, stock_weights, gated, rf, r
from core import *
X=pickle.load(open('exp2.pkl','rb')); G=X['G']
WF=('2019-01',DEV_END); HO=(HOLD_START,LAST_MONTH)
rows=[]
for univ in [300,500,1000]:
    for topn in [10,20,30,50]:
        W=stock_weights(univ,topn)['52WH']
        for gn in ['PxMA10','DD12>-5%','ALWAYS']:
            x=gated(W,G[gn],50)
            a=rmetrics(x.loc[WF[0]:WF[1]],rf); b=rmetrics(x.loc[HO[0]:HO[1]],rf)
            rows.append((univ,topn,gn,a['Sharpe'],a['MDD'],a['CAGR'],b['Sharpe'],b['MDD'],b['CAGR']))
R=pd.DataFrame(rows,columns=['univ','topn','gate','WF_Sh','WF_MDD','WF_CAGR','HO_Sh','HO_MDD','HO_CAGR'])
print(R.round(3).to_string())
# 52주 창 섭동
c=pd.read_parquet(f'{exp2.DATA}/panel_close.parquet')
from stocks import load_panels, adjusted_returns
cc,ss,vv,mm=load_panels(); ra,E=adjusted_returns(cc,ss); idx=(1+ra.fillna(0)).cumprod().where(cc.notna()); per=cc.index.to_period('M')
U=S['elig']&(S['rank']<=500); rows=[]
for L in [126,189,252,378,504]:
    h=(idx/idx.rolling(L,min_periods=int(L*0.8)).max()).groupby(per).last().reindex(U.index)
    sc=h.where(U&h.notna()); rr=sc.rank(axis=1,ascending=False,method='first'); w=(rr<=20).astype(float); w=w.div(w.sum(1).replace(0,np.nan),axis=0)
    for gn in ['PxMA10','ALWAYS']:
        x=gated(w,G[gn],50); a=rmetrics(x.loc[WF[0]:WF[1]],rf); b=rmetrics(x.loc[EVAL_START:DEV_END],rf)
        rows.append((L,gn,b['Sharpe'],a['Sharpe'],a['MDD']))
print(pd.DataFrame(rows,columns=['window_d','gate','DEV_Sh','WF_Sh','WF_MDD']).round(3).to_string())
# 연도별, 회전율
W=stock_weights()['52WH']; x=gated(W,G['PxMA10'],50); x0=gated(W,G['ALWAYS'],50)
yr=pd.DataFrame({'52WH|PxMA10':x,'52WH ungated':x0,'KOSPI':r,'UNIV-EW':gated(stock_weights()['UNIV-EW'],G['ALWAYS'],50)}).loc[EVAL_START:LAST_MONTH]
print(yr.groupby(yr.index.year).apply(lambda d:(1+d).prod()-1).round(3).T.to_string())
Wf=W.fillna(0); print('52WH monthly one-way turnover (ungated) avg', round(((Wf-Wf.shift(1)).abs().sum(1)/2).loc['2016':].mean(),3))
print('avg mcap rank of picks', round(S['rank'].where(W>0).stack().median(),1))
pickle.dump(dict(R=R,yr=yr),open('ev2c.pkl','wb'))
