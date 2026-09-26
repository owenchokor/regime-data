import pickle, numpy as np, pandas as pd, statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap, SPA
from exp2 import rmetrics, S
_G=pickle.load(open('exp2.pkl','rb'))['G']
from core import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
X=pickle.load(open('exp2.pkl','rb')); rf=X['rf']; r=X['r']; grid=X['grid']; ig=X['idxgrid']
cols=['n','CAGR','Vol','Sharpe','Sortino','MDD','Calmar','CVaR5','MaxUW','WinRate']
WF=('2019-01',DEV_END); HO=(HOLD_START,LAST_MONTH)
def T(d,a,b): return pd.DataFrame({k:rmetrics(v.loc[a:b],rf) for k,v in d.items()}).T[cols]
# 최종(사전 규칙): B = N[CAGR|MDD] → 동결 52WH|PxMA10 ; A = N[Sortino] → 동결 TermUp
FB=X['B']['freeze']['CAGR|MDD']; FA=X['A']['freeze']['Sortino']; print('frozen B',FB,'A',FA)
for c in [0,10,20,50]:
    d={'KOSPI B&H':r,f'B:{FB}':grid[c][FB],'B:52WH|DD12>-5% (참고)':grid[c]['52WH|DD12>-5%'],'B:52WH|ALWAYS (참고)':grid[c]['52WH|ALWAYS'],'UNIV-EW':grid[c]['UNIV-EW|ALWAYS'],f'A:{FA}':ig[c][FA]}
    print(f'== HOLDOUT cost {c}'); print(T(d,*HO).round(3))
def tests(x,y,label,nb=5000):
    x=x.dropna(); y=y.reindex(x.index); rfx=rf.reindex(x.index); ex=x-rfx; ey=y-rfx; n=len(x); lag=int(np.floor(4*(n/100)**(2/9)))
    ca=sm.OLS(ex.values,sm.add_constant(ey.values)).fit(cov_type='HAC',cov_kwds={'maxlags':lag})
    sh=lambda a: a.mean()/a.std(ddof=1)*np.sqrt(12)
    bs=StationaryBootstrap(6,ex.values,ey.values,seed=7); sd=[]
    for (a,b),_ in bs.bootstrap(nb): sd.append(sh(a)-sh(b))
    sd=np.array(sd); obs=sh(ex.values)-sh(ey.values)
    res=dict(n=n,alpha_ann=ca.params[0]*12,alpha_t=ca.tvalues[0],alpha_p=ca.pvalues[0],beta=ca.params[1],sh=sh(ex.values),bh=sh(ey.values),dsh=obs,lo=np.nanpercentile(sd,2.5),hi=np.nanpercentile(sd,97.5),p=np.nanmean(sd-np.nanmean(sd)>=obs))
    print(label,{k:round(v,3) for k,v in res.items()}); return res
ST={}
ST['B WF (nested)']=tests(X['B']['nested']['CAGR|MDD'][0].loc[WF[0]:WF[1]],r,'B WF nested')
ST['B HOLD']=tests(grid[50][FB].loc[HO[0]:HO[1]],r,'B HOLD')
ST['B WF+HOLD']=tests(pd.concat([X['B']['nested']['CAGR|MDD'][0].loc[WF[0]:WF[1]],grid[50][FB].loc[HO[0]:HO[1]]]),r,'B WF+HOLD')
ST['A WF (nested)']=tests(X['A']['nested']['Sortino'][0].loc[WF[0]:WF[1]],r,'A WF nested')
ST['A HOLD']=tests(ig[20][FA].loc[HO[0]:HO[1]],r,'A HOLD')
# 게이트 분해: 52WH 종목선택 자체 vs 게이트
ST['52WH ungated WF']=tests(grid[50]['52WH|ALWAYS'].loc[WF[0]:WF[1]],r,'52WH ungated WF')
ST['52WH ungated vs UNIV-EW WF']=tests(grid[50]['52WH|ALWAYS'].loc[WF[0]:WF[1]],grid[50]['UNIV-EW|ALWAYS'],'52WH vs EW WF')
# SPA: 개발구간, ML 제외 조합, 50bps, 벤치 KOSPI, 초과수익
L=grid[50][[k for k in grid[50] if not k.startswith('ML:')]].loc[EVAL_START:DEV_END].dropna(axis=1,how='any')
b=r.loc[L.index]; print('SPA grid',L.shape)
for nm,M in [('raw',L),('vol-matched',(L.sub(rf.loc[L.index],axis=0)).apply(lambda c:c*(b-rf.loc[L.index]).std()/c.std()).add(rf.loc[L.index],axis=0))]:
    sp=SPA(-b.values,-M.values,block_size=6,reps=5000,seed=7); sp.compute(); print('SPA',nm,np.round(sp.pvalues.values,3)); ST[f'SPA_{nm}']=sp.pvalues.values
# 무작위 포트폴리오: 같은 유니버스에서 20종목 무작위 + 같은 게이트(PxMA10, DD12>-5%) 
rng=np.random.default_rng(7); U=S['elig']&(S['rank']<=500); fwd=S['fwd'].fillna(0)
def randport(gname,a,b,k=500):
    g=_G[gname]; out=[]
    Uv=U.values; F=fwd.values; months=U.index
    for _ in range(k):
        ret=[]
        for i in range(len(months)):
            idx=np.where(Uv[i])[0]
            ret.append(F[i,rng.choice(idx,20,replace=False)].mean() if len(idx)>=20 else np.nan)
        gross=pd.Series(ret,index=months); gg=g.reindex(months)
        x=(gg*gross+(1-gg)*rf.shift(-1).reindex(months)).shift(1).reindex(r.index).loc[a:b]  # 비용 제외
        out.append(rmetrics(x,rf)['Sharpe'])
    return np.array(out)
for gname,key in [('PxMA10','52WH|PxMA10'),('DD12>-5%','52WH|DD12>-5%'),('ALWAYS','52WH|ALWAYS')]:
    for per,(a,b_) in [('WF',WF),('HOLD',HO)]:
        dist=randport(gname,a,b_,300); obs=rmetrics(grid[0][key].loc[a:b_],rf)['Sharpe']
        print(f'random-20 {gname} {per}: obs {obs:.2f} null mean {dist.mean():.2f} p={np.mean(dist>=obs):.3f}'); ST[f'rand_{gname}_{per}']=(obs,dist)
pickle.dump(ST,open('ev2_tests.pkl','wb'))
