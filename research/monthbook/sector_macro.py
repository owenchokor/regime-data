import pickle, numpy as np, pandas as pd
X = pickle.load(open("feats_ex.pkl", "rb")); F = X["F"].copy(); M = X["M"]
G = {"금융": ["금융", "기타금융", "보험", "은행", "증권"], "반도체·IT하드웨어": ["반도체", "IT부품", "전기·전자", "정보기기", "통신장비"],
     "SW·인터넷·콘텐츠": ["IT 서비스", "소프트웨어", "인터넷", "디지털컨텐츠", "컴퓨터서비스", "방송서비스", "오락·문화", "출판·매체복제", "통신", "통신서비스"],
     "소재": ["화학", "금속", "비금속", "종이·목재", "광업"], "산업재": ["기계·장비", "운송장비·부품", "건설", "운송·창고", "기타제조"],
     "소비": ["유통", "음식료·담배", "섬유·의류", "일반서비스", "농업 임업 및 어업"], "헬스케어": ["제약", "의료·정밀기기"], "유틸·부동산": ["전기·가스", "부동산"]}
inv = {s: g for g, ss in G.items() for s in ss}
F["grp"] = F.sector.map(inv).fillna("기타")
F["rel"] = F.ret - F.groupby("t").ret.transform("mean")
F["pctl"] = F.groupby("t").ret.rank(pct=True)
g = F.groupby("grp").agg(n=("rel", "size"), rel=("rel", "mean"), rel_med=("rel", "median"), top5_rate=("pctl", lambda x: (x > 0.75).mean()), bot5_rate=("pctl", lambda x: (x <= 0.25).mean()))
print(g.round(3).sort_values("rel").to_string())
F = F.join(M, on="t")
for reg, cond in {"금리상승(국고3Y 3M↑)": F.ktb3_3m_chg > 0, "금리하락": F.ktb3_3m_chg <= 0, "신용스프레드 확대": F.credit_3m_chg > 0, "신용스프레드 축소": F.credit_3m_chg <= 0,
                  "원화약세(3M)": F.usdkrw_3m > 0, "원화강세": F.usdkrw_3m <= 0, "KOSPI 3M 강세(>10%)": F.kospi_3m > 0.10, "KOSPI 3M 보통": F.kospi_3m <= 0.10,
                  "나스닥 3M↑": F.nq_3m > 0, "나스닥 3M↓": F.nq_3m <= 0}.items():
    sub = F[cond]; r = sub.groupby("grp").rel.mean()
    print(f"{reg:22s} 월수 {sub.t.nunique():3d} | " + " ".join(f"{k}:{v*100:+.1f}" for k, v in r.sort_values().items()))
pickle.dump(F, open("F_grp.pkl", "wb"))
