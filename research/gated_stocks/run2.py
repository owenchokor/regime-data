import pickle, numpy as np, pandas as pd, time
from exp2 import *
t0=time.time()
G=gates(); ONE=pd.Series(1.0,index=r.index)
# (A) 지수
fam=rule_grid(f); IDX={k:v for d_ in fam.values() for k,v in d_.items()}
IDX.update({k:G[k] for k in ['TermUp','CredEasing','RateFalling']}); IDX['B&H']=ONE
ir=lambda k,c: idx_returns(IDX[k],c)
A={'nested':{},'freeze':{}}
for obj in OBJECTIVES:
    A['nested'][obj]=nested(IDX,ir,obj,IDX_SEL_COST,r)
    A['freeze'][obj]=freeze(IDX,ir,obj,IDX_SEL_COST,r)
print('A done',round(time.time()-t0))
# (B) 개별주
W=stock_weights(); W.update(ml_weights()); print('weights',len(W),round(time.time()-t0))
combos={f'{s}|{g}':(s,g) for s in W for g in G}
nest_c={k:v for k,v in combos.items() if not k.startswith('ML:')}  # ML은 훈련구간 이력이 없어 nested 후보에서 제외
cache={}
def sr(k,c):
    key=(k,c)
    if key not in cache: s,g=combos[k]; cache[key]=gated(W[s],G[g],c)
    return cache[key]
B={'nested':{},'freeze':{}}
for obj in OBJECTIVES:
    B['nested'][obj]=nested(nest_c,sr,obj,STOCK_SEL_COST,r)
    B['freeze'][obj]=freeze(nest_c,sr,obj,STOCK_SEL_COST,r)
    print(obj,B['nested'][obj][1],'freeze',B['freeze'][obj])
grid={c:pd.DataFrame({k:sr(k,c) for k in combos}) for c in [0,10,20,50]}
idxgrid={c:pd.DataFrame({k:ir(k,c) for k in IDX}) for c in [0,10,20,50]}
pickle.dump(dict(A=A,B=B,grid=grid,idxgrid=idxgrid,combos=combos,G=G,IDX=IDX,rf=rf,r=r),open('exp2.pkl','wb'))
print('done',round(time.time()-t0))
