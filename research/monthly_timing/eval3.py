import pickle, numpy as np, pandas as pd, statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap, SPA
from core import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
R=pickle.load(open('wf.pkl','rb')); r=R['r']; m=R['m']; fam=R['fam']; f=R['f']
ONE=pd.Series(1.0,index=r.index)
FINAL=fam['DD']['DD12>-5%']                 # 개발구간 동결 결과
WFPOS=R['nested'][('DD','expanding')][0]     # walk-forward OOS 포지션(= 동일 파라미터)
periods={'WF-OOS':('2019-01',DEV_END),'HOLDOUT':(HOLD_START,LAST_MONTH),'DEV(in-sample)':(EVAL_START,DEV_END),'FULL':(EVAL_START,LAST_MONTH)}
out={}
cols=['n','CAGR','Cum','Vol','Sharpe','MDD','Calmar','WinRate','DownDev','Worst','Best','Exposure','TurnoverYr']
for nm,(a,b) in periods.items():
    pos = WFPOS if nm=='WF-OOS' else FINAL
    t={}
    for c in COSTS_BPS:
        t[f'DD_{c}bp']=metrics(strat_returns(pos,r,c).loc[a:b],pos)
    t['B&H']=metrics(r.loc[a:b],ONE)
    T=pd.DataFrame(t).T[cols]; out[nm]=T
    bh=T.loc['B&H']; s=T.loc['DD_20bp']
    print(f'== {nm} {a}~{b}'); print(T.round(3))
    print('  excess CAGR %.3f  Sharpe+ %.3f  MDD red %.3f  Calmar+ %.3f  DownDev red %.3f' % (s.CAGR-bh.CAGR, s.Sharpe-bh.Sharpe, bh.MDD-s.MDD, s.Calmar-bh.Calmar, bh.DownDev-s.DownDev))
pickle.dump(out,open('final_tables.pkl','wb'))

def tests(a,b,pos,label,nb=5000):
    x=strat_returns(pos,r,20).loc[a:b].values; y=r.loc[a:b].values; d=x-y; n=len(d)
    lag=int(np.floor(4*(n/100)**(2/9)))
    nw=sm.OLS(d,np.ones(n)).fit(cov_type='HAC',cov_kwds={'maxlags':lag})
    nws=sm.OLS(x,np.ones(n)).fit(cov_type='HAC',cov_kwds={'maxlags':lag})
    # 위험조정: 전략 = a + b*시장 (시장타이밍 알파)
    ca=sm.OLS(x,sm.add_constant(y)).fit(cov_type='HAC',cov_kwds={'maxlags':lag})
    bs=StationaryBootstrap(6,x,y,seed=7)
    sd=[];ss=[]
    for (xb,yb),_ in bs.bootstrap(nb):
        ss.append(sharpe(xb)); sd.append(sharpe(xb)-sharpe(yb))
    sd=np.array(sd); ss=np.array(ss)
    obs=sharpe(x)-sharpe(y)
    # 중심화 부트스트랩 p (H0: ΔSharpe<=0)
    p_boot=np.mean(sd-sd.mean()>=obs)
    # 순열: 포지션 시계열 원형 이동(노출·신호 자기상관 보존, 수익과의 정렬만 파괴)
    pp=pos.shift(1).loc[a:b].values; rng=np.random.default_rng(7); perm=[]
    for k in range(1,n):
        xs=np.roll(pp,k)*y - np.abs(np.diff(np.r_[np.roll(pp,k)[0],np.roll(pp,k)]))*20/1e4
        perm.append(sharpe(xs))
    p_perm=np.mean(np.array(perm)>=sharpe(x))
    print(f'-- {label} n={n} NWlag={lag}')
    print(f'   mean diff/mo {d.mean():.4f}  NW t={nw.tvalues[0]:.2f} p={nw.pvalues[0]:.3f} | strat mean NW t={nws.tvalues[0]:.2f}')
    print(f'   timing alpha/mo {ca.params[0]:.4f} t={ca.tvalues[0]:.2f} p={ca.pvalues[0]:.3f} beta={ca.params[1]:.2f}')
    print(f'   Sharpe strat {sharpe(x):.3f} 95%CI [{np.percentile(ss,2.5):.2f},{np.percentile(ss,97.5):.2f}]  B&H {sharpe(y):.3f}')
    print(f'   dSharpe {obs:.3f} 95%CI [{np.percentile(sd,2.5):.2f},{np.percentile(sd,97.5):.2f}] p_boot={p_boot:.3f}  p_perm(circular,{n-1})={p_perm:.3f}')
    return dict(n=n,diff=d.mean(),nw_t=nw.tvalues[0],nw_p=nw.pvalues[0],alpha=ca.params[0],alpha_t=ca.tvalues[0],alpha_p=ca.pvalues[0],beta=ca.params[1],
                sh=sharpe(x),sh_lo=np.percentile(ss,2.5),sh_hi=np.percentile(ss,97.5),bh=sharpe(y),dsh=obs,dsh_lo=np.percentile(sd,2.5),dsh_hi=np.percentile(sd,97.5),p_boot=p_boot,p_perm=p_perm,perm=np.array(perm))
T={}
T['WF-OOS']=tests('2019-01',DEV_END,WFPOS,'WF-OOS')
T['HOLDOUT']=tests(HOLD_START,LAST_MONTH,FINAL,'HOLDOUT')
T['WF+HOLD']=tests('2019-01',LAST_MONTH,FINAL,'WF-OOS+HOLDOUT (same rule)')
pickle.dump(T,open('tests.pkl','wb'))

# SPA / Reality Check: 개발구간 고정 파라미터 룰 그리드 전체
flat={k:v for d in fam.values() for k,v in d.items()}
L=pd.DataFrame({k:strat_returns(p,r,20) for k,p in flat.items()}).loc[EVAL_START:DEV_END].dropna()  # TSMOM24 첫 달 결측 → 행 제거
bench=r.loc[L.index]
print('grid size',L.shape)
for nm,M in [('raw return',L),('vol-matched',L.apply(lambda c: c*bench.std()/c.std()))]:
    spa=SPA(-bench.values,-M.values,block_size=6,reps=5000,bootstrap='stationary',seed=7); spa.compute()
    print(f'SPA [{nm}] pvalues lower/consistent/upper(=RC):',np.round(spa.pvalues.values,3))
# 개별 룰 p (vol-matched 차이) + Holm
from statsmodels.stats.multitest import multipletests
pv={}
for k in L:
    d=(L[k]*bench.std()/L[k].std()-bench).values
    pv[k]=sm.OLS(d,np.ones(len(d))).fit(cov_type='HAC',cov_kwds={'maxlags':4}).pvalues[0]/2*(1 if d.mean()>0 else -1)+ (0 if d.mean()>0 else 1)
pv=pd.Series(pv).sort_values()
rej=multipletests(pv.values,method='holm',alpha=0.05)[0]
print('top raw one-sided p:',pv.head(5).round(4).to_dict(),' Holm rejections:',rej.sum())
