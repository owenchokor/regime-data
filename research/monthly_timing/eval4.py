import pickle, numpy as np, pandas as pd
from core import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
R=pickle.load(open('wf.pkl','rb')); r=R['r']; m=R['m']; fam=R['fam']; f=R['f']
ONE=pd.Series(1.0,index=r.index); FINAL=fam['DD']['DD12>-5%']
# A. 다음달 수익률 예측력 (WF-OOS)
a,b='2019-01',DEV_END; y=r.loc[a:b]
cand=pickle.load(open('cand.pkl','rb'))
print('always-up hit', round((y>0).mean(),3))
for k in ['ML:Logit_bin:exp','ML:Logit_bin:rol','ML:RF_bin:exp','ML:GBM_bin:rol','ML:LGBM_bin:exp','ML:OLS_bin:exp','ML:Ridge_bin:exp','N:DD:exp','Committee8']:
    p=cand[k].shift(1).loc[a:b]; pb=(p>0.5)
    print(k,'hit',round(((pb)==(y>0)).mean(),3),'P(up|in)',round((y[pb]>0).mean(),3),'P(up|out)',round((y[~pb]>0).mean(),3),'mean r|in',round(y[pb].mean(),4),'mean r|out',round(y[~pb].mean(),4))
# OOS R2 (Campbell-Thompson) — 회귀 예측 재계산
from strategies import _prune_corr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
X_all=f[ML_FEATS]; yn=r.shift(-1)
pr={'OLS':[], 'Ridge':[], 'HistMean':[]}; act=[]
for yr in WF_YEARS:
    te=f'{yr-1}-12'; rows=X_all.loc['2015-12':str(pd.Period(te)-1)].dropna(); yv=yn.loc[rows.index]
    cols=_prune_corr(rows); mu=rows[cols].mean(); sd=rows[cols].std()
    dec=pd.period_range(pd.Period(f'{yr}-01')-1, pd.Period(DEV_END if yr==2024 else f'{yr}-12')-1,freq='M')
    Xo=((X_all.loc[dec,cols]-mu)/sd).values; Xt=((rows[cols]-mu)/sd).values
    pr['OLS']+=list(LinearRegression().fit(Xt,yv).predict(Xo)); pr['Ridge']+=list(RidgeCV(alphas=np.logspace(-2,3,20),cv=TimeSeriesSplit(3)).fit(Xt,yv).predict(Xo))
    pr['HistMean']+=[yv.mean()]*len(dec); act+=list(yn.loc[dec].values)
act=np.array(act)
for k in ['OLS','Ridge']:
    print('OOS R2',k, round(1-np.sum((act-np.array(pr[k]))**2)/np.sum((act-np.array(pr['HistMean']))**2),3))
# B. 서브기간 (최종 룰 고정; 2016-2018,2019-21,2022-24H1 = 개발구간 내부 → in-sample 표시)
print('== subperiods (20bps)')
for nm,(s0,s1) in {'2016-2018(IS)':('2016-01','2018-12'),'2019-2021(WF)':('2019-01','2021-12'),'2022-2024H1(WF)':('2022-01',DEV_END),'2024H2-2026(HOLD)':(HOLD_START,LAST_MONTH)}.items():
    x=strat_returns(FINAL,r,20).loc[s0:s1]; bh=r.loc[s0:s1]; ms=metrics(x,FINAL); mb=metrics(bh)
    print(f'{nm:20s} DD CAGR {ms["CAGR"]:.3f} Sh {ms["Sharpe"]:.2f} MDD {ms["MDD"]:.3f} exp {ms["Exposure"]:.2f} | B&H CAGR {mb["CAGR"]:.3f} Sh {mb["Sharpe"]:.2f} MDD {mb["MDD"]:.3f}')
# C. 레짐 (월초 기준 과거 정보로 분류, 2016-01~2026-08)
x=strat_returns(FINAL,r,20).loc[EVAL_START:LAST_MONTH]; bh=r.loc[x.index]
bull=(f['mom12'].shift(1).loc[x.index]>0); hv=(f['vol6'].shift(1).loc[x.index]>f['vol6'].shift(1).expanding(12).median().loc[x.index])
reg=pd.DataFrame({'x':x,'bh':bh,'bull':bull,'hv':hv})
for c,lab in [('bull','bull(12M>0)'),('hv','highvol(6M>med)')]:
    for v in [True,False]:
        g=reg[reg[c]==v]; print(f'{lab}={v}: n={len(g)} DD mean {g.x.mean():.4f} B&H mean {g.bh.mean():.4f} diff {g.x.mean()-g.bh.mean():.4f} | DD vol {g.x.std()*np.sqrt(12):.3f} B&H vol {g.bh.std()*np.sqrt(12):.3f}')
# D. 위기
print('== crisis windows (cum return)')
for nm,(s0,s1) in {'2018-10~2019-01':('2018-10','2019-01'),'COVID 2020-02~2020-04':('2020-02','2020-04'),'2021-07~2022-09 bear':('2021-07','2022-09'),'2024-07~2024-12':('2024-07','2024-12'),'2026-03':('2026-03','2026-03'),'2026-07':('2026-07','2026-07')}.items():
    xx=strat_returns(FINAL,r,20).loc[s0:s1]; bb=r.loc[s0:s1]; pos=FINAL.shift(1).loc[s0:s1]
    print(f'{nm:25s} DD {(1+xx).prod()-1:+.3f}  B&H {(1+bb).prod()-1:+.3f}  pos {list(pos.astype(int).values)}')
# E. 실행지연 1거래일
d=load_daily(); fd=d.groupby(d.index.to_period('M')).head(1)  # 각 월 첫 거래일
fd.index=fd.index.to_period('M'); fd=fd.loc[:LAST_MONTH]
r1=fd['kospi'].pct_change().shift(-1)  # t+1월 첫거래일→t+2월 첫거래일 = 'r1[t+1]'
r1=fd['kospi'].pct_change()  # 인덱스 t: (t월 첫거래일 / t-1월 첫거래일)
# pos[t] 결정 → t+1월 첫거래일 체결 → t+2월 첫거래일까지 보유: 수익 인덱스 t+2
xs=(FINAL.shift(2)*r1 - FINAL.shift(2).diff().abs()*20/1e4)
for nm,(s0,s1) in {'WF':('2019-02',DEV_END),'HOLD':(HOLD_START,LAST_MONTH)}.items():
    a_=metrics(xs.loc[s0:s1]); b_=metrics(r1.loc[s0:s1]); print(f'lag1d {nm}: DD Sh {a_["Sharpe"]:.3f} MDD {a_["MDD"]:.3f} | B&H Sh {b_["Sharpe"]:.3f} MDD {b_["MDD"]:.3f}')
# F. 탐색 규모
n_grid=sum(len(v) for v in fam.values()); n_nested=len(R['nested']); n_ml=sum(len(o) for o,_ in R['ml'].values())
print('grid',n_grid,'nested',n_nested,'ml',n_ml,'committee 1, baselines 3, perturb',9*5+11, 'total WF-OOS configs', n_nested+n_ml+1)
# G. 피처 상관
C=f[ML_FEATS].loc['2015-12':DEV_END].corr(); print(C.round(2))
pickle.dump(dict(C=C,reg=reg),open('eval4.pkl','wb'))
