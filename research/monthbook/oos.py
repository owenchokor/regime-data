import pickle, numpy as np, pandas as pd
from ic import ic_table
F = pickle.load(open("F_all.pkl", "rb"))
# 왜: monthbook 표는 사후 수익순 정렬 → 동점 처리 시 순서가 사후정보를 누설. 티커순으로 재정렬해 차단
F = F.sort_values(["t", "ticker"]).reset_index(drop=True); book = pickle.load(open("monthbook.pkl", "rb"))
FEATS = ["r1m","r3m","r6m","r12m","dist_low","h52","trend_age","newhi20","vol60","vol20","dd60","turn_ratio","amt_ratio","log_amt","log_mcap","crowd","EP","BP","PER","PBR","DIV","loss","eps_g12","eps_g3","bps_g12"]
F["yr"] = [p.year for p in F.t]
def pct(f): return F.groupby("t")[f].rank(pct=True).fillna(0.5)
def score(signs): return sum(s * (pct(f) - 0.5) for f, s in signs.items())
def evaluate(sc, months, k=5):
    rows = []
    for t in months:
        g = F[F.t == t].assign(sc=sc[F.t == t])
        ic = g.sc.rank().corr(g.ret.rank())
        top = g.nlargest(k, "sc").ret.mean(); pool = g.ret.mean(); bot = g.nsmallest(k, "sc").ret.mean()
        rows.append(dict(t=t, ic=ic, top=top, pool=pool, bot=bot, kospi=book[t]["kospi"].iloc[-1]))
    return pd.DataFrame(rows).set_index("t")
def summ(E):
    n = len(E); cagr = lambda x: (1 + x).prod() ** (12 / n) - 1
    return dict(월수=n, IC=E.ic.mean(), IC_t=E.ic.mean() / E.ic.std() * np.sqrt(n), top5_월평균=E.top.mean(), 후보20_월평균=E.pool.mean(), bot5_월평균=E.bot.mean(),
                top5이_후보평균_상회=(E.top > E.pool).mean(), top5_연환산=cagr(E.top), 후보20_연환산=cagr(E.pool), KOSPI_연환산=cagr(E.kospi))
first = F[F.yr <= 2020]; T1 = ic_table(first, FEATS)
sel = T1[(T1.t.abs() >= 1.5) & (T1.n >= 15)]
oos_signs = {f: np.sign(r.IC) for f, r in sel.iterrows()}
print("전반(2016-20)에서 선택된 피처:", {f: int(s) for f, s in oos_signs.items()})
months2 = sorted(F[F.yr >= 2021].t.unique()); months1 = sorted(F[F.yr <= 2020].t.unique())
E_oos = evaluate(score(oos_signs), months2)
FINAL = {"eps_g3": 1, "eps_g12": 1, "amt_ratio": -1, "r1m": -1, "vol20": -1, "h52": -1}
sc_final = score(FINAL)
E_f1 = evaluate(sc_final, months1); E_f2 = evaluate(sc_final, months2); E_all = evaluate(sc_final, sorted(F.t.unique()))
R = pd.DataFrame({"표본외(전반 선택→후반)": summ(E_oos), "최종규칙-전반": summ(E_f1), "최종규칙-후반": summ(E_f2), "최종규칙-전체": summ(E_all)})
pd.set_option("display.width", 220); print(R.round(3).to_string())
# 제외 규칙 효과: 딜고정 proxy (20일 변동성 하위 10% & h52>=0.99), 적자, 자본잠식
F["deal_proxy"] = ((F.groupby("t").vol20.rank(pct=True) <= 0.10) & (F.h52 >= 0.99)).astype(int)
for nm, m in {"딜고정 proxy": F.deal_proxy == 1, "적자(EPS<=0)": F.loss == 1, "자본잠식(BPS<=0)": F.impair == 1, "거래대금비 상위25%": F.groupby("t").amt_ratio.rank(pct=True) > 0.75, "1M 상위25%": F.groupby("t").r1m.rank(pct=True) > 0.75, "EPS 3M 상향": F.eps_g3 > 0.0001, "EPS 12M 증가": F.eps_g12 > 0}.items():
    rel = (F.ret - F.groupby("t").ret.transform("mean"))[m]
    worst = (F.groupby("t").ret.rank(pct=True) <= 0.25)[m]
    print(f"{nm:16s} n={m.sum():4d} 월내 상대수익 평균 {rel.mean()*100:+.2f}%p 중앙 {rel.median()*100:+.2f}%p 하위25% 비율 {worst.mean():.2f}")
F["sc_final"] = sc_final; F["rk_final"] = F.groupby("t").sc_final.rank(ascending=False, method="first")
pickle.dump(dict(F=F, E_oos=E_oos, E_f1=E_f1, E_f2=E_f2, E_all=E_all, R=R, oos_signs=oos_signs, T1=T1), open("oos.pkl", "wb"))
