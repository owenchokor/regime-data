import pickle, numpy as np, pandas as pd
from exp2 import rmetrics, rf, r
from core import *
WF=('2019-01',DEV_END); HO=(HOLD_START,LAST_MONTH)
D=pickle.load(open('run3.pkl','rb')); B=pickle.load(open('run3b.pkl','rb')); H=pickle.load(open('figs3_html.pkl','rb'))
V={k:v[0] for k,v in D['res'].items()}; V['V7 업종 상한(업종당 최대 4)']=B['cap4']
nb={('V4',.10):D['nb'][('V4 종목 트레일링 스톱 15%',"{'stock_stop': 0.1}")],('V4',.20):D['nb'][('V4 종목 트레일링 스톱 15%',"{'stock_stop': 0.2}")]}
b=rmetrics(V['BASE'].loc[WF[0]:WF[1]],rf)
NBS={'V1 과열제외(1M 상위10%)':[("{'exclude_hot': 0.05}"),("{'exclude_hot': 0.2}")],'V4 종목 트레일링 스톱 15%':["{'stock_stop': 0.1}","{'stock_stop': 0.2}"],'V5 포트폴리오 스톱 10%':["{'port_stop': 0.07}","{'port_stop': 0.15}"],'V6 일간 게이트(200일선)':["{'daily_gate': 120}","{'daily_gate': 250}"]}
rows=[]
for k,x in V.items():
    m=rmetrics(x.loc[WF[0]:WF[1]],rf); h=rmetrics(x.loc[HO[0]:HO[1]],rf)
    c1=m['Sharpe']>=b['Sharpe']-0.05; c2=(m['MDD']-b['MDD']>=0.03) or (m['CVaR5']-b['CVaR5']>=0.03)
    if k in NBS: c3=all(rmetrics(D['nb'][(k,o)].loc[WF[0]:WF[1]],rf)['Sharpe']>=b['Sharpe']-0.05 for o in NBS[k])
    elif k.startswith('V7'): c3=all(rmetrics(B[f'cap{c}'].loc[WF[0]:WF[1]],rf)['Sharpe']>=b['Sharpe']-0.05 for c in (3,6))
    else: c3=None
    ok='기준' if k=='BASE' else ('통과' if (c1 and c2 and c3 is not False) else '탈락')
    mk=lambda z:'–' if k=='BASE' or z is None else ('○' if z else '✕')
    rows.append([k,f"{m['CAGR']*100:.1f}%",f"{m['Sharpe']:.2f}",f"{m['MDD']*100:.1f}%",f"{m['CVaR5']*100:.1f}%",mk(c1),mk(c2),mk(c3),ok,f"{h['Sharpe']:.2f}",f"{h['MDD']*100:.1f}%"])
T=pd.DataFrame(rows,columns=['변형','WF CAGR','WF Sharpe','WF MDD','WF CVaR5','①Sharpe','②위험개선','③이웃값','판정','Holdout Sharpe(진단)','Holdout MDD(진단)']).to_html(index=False,classes='t',border=0)
tpl=open('report3_tpl.html').read().replace('{{T}}',T)
for k,v in H.items(): tpl=tpl.replace('{{F_'+k+'}}',f'<div class="fig">{v}</div>')
open('52wh_variants_report.html','w').write(tpl); print('ok',tpl.count('{{')); print(pd.read_html(T)[0].to_string())
