"""52WH 후보 20개 최종 리포트: 6관점 사례분석(74개월) + 종합 결론. 사전정보만 사용, 사후 수익은 채점용."""
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, CondPageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from news import NEWS, THEME
from timeline import TIMELINE
from ic import ic_table

FP = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"; FB = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
fm.fontManager.addfont(FP); fm.fontManager.addfont(FB)
plt.rcParams["font.family"] = fm.FontProperties(fname=FP).get_name(); plt.rcParams["axes.unicode_minus"] = False
pdfmetrics.registerFont(TTFont("NG", FP)); pdfmetrics.registerFont(TTFont("NGB", FB))
H1 = ParagraphStyle("h1", fontName="NGB", fontSize=15, leading=19, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="NGB", fontSize=10.5, leading=14, spaceAfter=3)
BD = ParagraphStyle("bd", fontName="NG", fontSize=8.6, leading=12.4, spaceAfter=3)
NR = ParagraphStyle("nr", fontName="NG", fontSize=7.6, leading=10.4, spaceAfter=1.5)
SM = ParagraphStyle("sm", fontName="NG", fontSize=6.5, leading=8)
OUT = "/home/claude/book"; cmap = plt.get_cmap("RdYlGn_r")

book = pickle.load(open("monthbook.pkl", "rb")); M = pickle.load(open("feats_ex.pkl", "rb"))["M"]
O = pickle.load(open("oos.pkl", "rb")); F = O["F"]; R = O["R"]
F["pct_ret"] = F.groupby("t").ret.rank(pct=True)
LAB = {"r1m": "1개월 상승률", "r12m": "12개월 상승률", "amt_ratio": "거래대금 급증도(20/120일)", "vol20": "단기 변동성", "h52": "52주 고점 근접도",
       "PER": "PER", "PBR": "PBR", "DIV": "배당수익률", "eps_g12": "EPS 전년비", "eps_g3": "EPS 3개월 변화", "log_mcap": "시가총액", "trend_age": "고점권 체류일수"}
PERSP = {"r1m": "시장분위기", "r12m": "시장분위기", "amt_ratio": "시장분위기", "vol20": "시장분위기", "h52": "시장분위기", "trend_age": "시장분위기",
         "turn_ratio": "시장분위기", "r3m": "시장분위기", "r6m": "시장분위기", "dist_low": "시장분위기", "newhi20": "시장분위기", "vol60": "시장분위기", "dd60": "시장분위기",
         "crowd": "시장분위기", "log_amt": "유동성·규모", "log_mcap": "유동성·규모", "PER": "가치평가·기대치", "PBR": "가치평가", "EP": "가치평가", "BP": "가치평가",
         "DIV": "가치평가", "loss": "재무", "eps_g12": "성장성", "eps_g3": "기대치(실적 상향)", "bps_g12": "성장성"}
PCOL = {"시장분위기": "#e6550d", "가치평가": "#3182bd", "가치평가·기대치": "#6baed6", "재무": "#756bb1", "성장성": "#31a354", "기대치(실적 상향)": "#74c476", "유동성·규모": "#969696"}
FEATS = list(PERSP)


def ic_chart(T, path):
    T = T.sort_values("IC"); se = (T.IC / T.t).abs().fillna(0)
    fig, ax = plt.subplots(figsize=(11.2, 4.4), dpi=130)
    ax.barh([f"{f} ({PERSP[f]})" for f in T.index], T.IC, xerr=1.96 * se, color=[PCOL[PERSP[f]] for f in T.index], ecolor="grey", capsize=2)
    ax.axvline(0, color="black", lw=0.6); ax.set_xlabel("월내 순위 IC (평균 ± 95% 구간)"); ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def oos_chart(path):
    e, a = O["E_oos"], O["E_all"]; a2 = a.loc[e.index]
    fig, ax = plt.subplots(figsize=(11.2, 3.3), dpi=130)
    x = [p.to_timestamp() for p in e.index]
    for s, nm, st in [(e.top, "표본외 규칙 상위5 (2016–20에서 선택)", "-"), (a2.top, "최종 규칙 상위5 (전체표본 정의, 참고)", "-."), (e.pool, "후보 20개 전체", "-"), (e.kospi, "KOSPI (같은 달)", "--")]:
        ax.plot(x, (1 + s).cumprod(), st, lw=1.6, label=nm)
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False); ax.set_ylabel("누적(게이트 ON 월만 연결)"); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def month_chart(b, path):
    fig, ax = plt.subplots(figsize=(11.2, 2.6), dpi=120)
    P = b["path"]; n = P.shape[1]; tb = b["table"]
    for i, c in enumerate(P.columns):
        ax.plot(P.index, P[c] * 100, color=cmap(i / (n - 1)), lw=1.1 if i in (0, n - 1) else 0.7)
    ax.plot(P.index, P.mean(axis=1) * 100, color="#1f4e9c", lw=2.2, label="후보 20개 평균")
    ax.plot(b["kospi"].index, b["kospi"] * 100, color="black", lw=1.5, ls="--", label="KOSPI")
    for i in (0, n - 1):
        ax.annotate(f"{i+1}. {tb.name.iloc[i]}", (P.index[-1], P.iloc[-1, i] * 100), fontsize=6.5, xytext=(4, 0), textcoords="offset points", va="center")
    ax.axhline(0, color="grey", lw=0.6); ax.set_ylabel("누적수익(%)", fontsize=7); ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=6.5, frameon=False); ax.tick_params(labelsize=6.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def trunc(s, n):
    s = "" if not isinstance(s, str) else s.strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def fmt_pct(x, d=0):
    return "–" if pd.isna(x) else f"{x*100:+.{d}f}%"


def month_table(t):
    tb = book[t]["table"]; g = F[F.t == t].set_index("ticker")
    head = ["순위", "종목", "KRX 업종", "한 줄 설명 (주요제품 / 산업분류)", "1M", "거래대금비", "PER", "EPS 전년비", "규칙순위", "보유수익", "MDD"]
    data = [head]
    for _, r in tb.iterrows():
        x = g.loc[r.ticker]
        desc = trunc(r.products, 40) or "–"; ind = trunc(r.industry, 22)
        line = (f"{desc} / {ind}" if ind else desc); line = ("[상폐] " if r.delisted else "") + line
        per = "적자" if x.loss == 1 else ("–" if pd.isna(x.PER) else f"{x.PER:.1f}")
        data.append([str(r["rank"]), trunc(r["name"], 11), trunc(r.sector, 8), Paragraph(line, SM), fmt_pct(x.r1m), "–" if pd.isna(x.amt_ratio) else f"{x.amt_ratio:.1f}x",
                     per, fmt_pct(x.eps_g12), str(int(x.rk_final)), f"{r.ret*100:+.1f}%", f"{r.mdd*100:.1f}%"])
    T = Table(data, colWidths=[8*mm, 22*mm, 17*mm, 112*mm, 14*mm, 16*mm, 13*mm, 17*mm, 14*mm, 17*mm, 15*mm], rowHeights=[5*mm] + [4.1*mm]*len(tb), repeatRows=1)
    st = [("FONT", (0, 0), (-1, 0), "NGB", 6.8), ("FONT", (0, 1), (-1, -1), "NG", 6.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e9c")),
          ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ALIGN", (4, 0), (-1, -1), "RIGHT"), ("ALIGN", (0, 0), (0, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")), ("TOPPADDING", (0, 0), (-1, -1), 0.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.4)]
    for i, (_, r) in enumerate(tb.iterrows(), 1):
        c = cmap((i - 1) / (len(tb) - 1)); st.append(("TEXTCOLOR", (9, i), (9, i), colors.Color(c[0]*0.8, c[1]*0.8, c[2]*0.8)))
        if g.loc[r.ticker].rk_final <= 5: st.append(("BACKGROUND", (8, i), (8, i), colors.HexColor("#fff3bf")))
    T.setStyle(TableStyle(st)); return T


def regime(m):
    rate = "금리 상승" if m.ktb3_3m_chg > 0.10 else ("금리 하락" if m.ktb3_3m_chg < -0.10 else "금리 보합")
    cred = "신용 경계(스프레드 확대)" if m.credit_3m_chg > 0.05 else ("신용 완화" if m.credit_3m_chg < -0.05 else "신용 안정")
    fx = "원화 약세" if m.usdkrw_3m > 0.02 else ("원화 강세" if m.usdkrw_3m < -0.02 else "환율 보합")
    mk = "지수 과열권(10개월선 +15% 초과)" if m.kospi_vs_ma10 > 0.15 else ("추세 초입(10개월선 +5% 이내)" if m.kospi_vs_ma10 < 0.05 else "상승 추세 중반")
    return mk, rate, cred, fx


def narrative(t):
    m = M.loc[t]; g = F[F.t == t].copy(); tb = book[t]["table"]; b = book[t]
    mk, rate, cred, fx = regime(m)
    macro = (f"<b>거시</b> · {mk}, {rate}, {cred}, {fx}. KOSPI 1M {fmt_pct(m.kospi_1m,1)} / 3M {fmt_pct(m.kospi_3m,1)} / 12M {fmt_pct(m.kospi_12m,1)}, "
             f"CD {m.cd:.2f}%(3M {m.cd_3m_chg*100:+.0f}bp), 국고3Y 3M {m.ktb3_3m_chg*100:+.0f}bp, 신용스프레드 {m.credit:.2f}%p(3M {m.credit_3m_chg*100:+.0f}bp), "
             f"원/달러 3M {fmt_pct(m.usdkrw_3m,1)}, 미10Y 3M {m.us10_3m_chg*100:+.0f}bp, 나스닥 3M {fmt_pct(m.nq_3m,1)}. (ECOS 금리는 공표 지연을 고려해 1개월 전 값)")
    grp = g.grp.value_counts(); top = ", ".join(f"{k} {v}" for k, v in grp.head(3).items())
    val = (f"<b>가치평가·재무</b> · 후보 PER 중앙값 {g.PER.median():.1f}배, PBR 중앙값 {g.PBR.median():.2f}배, 배당수익률 중앙값 {g.DIV.median():.1f}%, 적자 기업 {int((g.loss == 1).sum())}개. "
           f"업종 구성: {top} (최대 쏠림 {int(g.crowd.max())}개).")
    mood = (f"<b>시장분위기·기대치</b> · 결정일 기준 1개월 상승률 중앙값 {fmt_pct(g.r1m.median())}, 거래대금 급증도 중앙값 {g.amt_ratio.median():.1f}배, "
            f"52주 고점 정확 도달 {int((g.h52 >= 0.999).sum())}개, 최근 3개월 EPS 상향 {int((g.eps_g3 > 1e-4).sum())}개.")
    grow = f"<b>성장성</b> · EPS 전년비 증가 {int((g.eps_g12 > 0).sum())}개 / 감소 {int((g.eps_g12 < 0).sum())}개, EPS 전년비 중앙값 {fmt_pct(g.eps_g12.median())}."
    gi = g.set_index("ticker")
    def who(r):
        x = gi.loc[r.ticker]
        per = "적자" if x.loss == 1 else (f"PER {x.PER:.0f}" if pd.notna(x.PER) else "PER –")
        return f"{r['name']}({r.ret*100:+.0f}%; 1M {fmt_pct(x.r1m)}, 거래대금 {x.amt_ratio:.1f}x, {per}, EPS {fmt_pct(x.eps_g12)})"
    res = (f"<b>결과</b> · 후보 평균 {fmt_pct(tb.ret.mean(),1)} vs KOSPI {fmt_pct(b['kospi'].iloc[-1],1)}. 최선: " + "; ".join(who(r) for _, r in tb.head(3).iterrows()) +
           ". 최악: " + "; ".join(who(r) for _, r in tb.tail(3).iloc[::-1].iterrows()) + ".")
    best = g.nlargest(5, "ret").ticker; worst = g.nsmallest(5, "ret").ticker
    P = pd.DataFrame({f: g[f].rank(pct=True).values for f in LAB}, index=g.ticker.values)
    d = (P.loc[best].mean() - P.loc[worst].mean()).dropna()
    d = d[[f for f in d.index if P[f].notna().sum() > 10 and P[f].nunique() > 3]]
    d = d.reindex(d.abs().sort_values(ascending=False).index).head(3)
    diffs = [f"{LAB[f]}({PERSP[f]}) — 최선군 백분위 {P.loc[best, f].mean()*100:.0f} vs 최악군 {P.loc[worst, f].mean()*100:.0f}, 최선군이 {'높음' if v > 0 else '낮음'}" for f, v in d.items()]
    disc = "<b>판별 요인(사전정보)</b> · " + ("; ".join(diffs) if diffs else "뚜렷한 차이 없음") + "."
    top5 = g.nsmallest(5, "rk_final"); tr = top5.ret.mean(); pr = g.ret.mean(); pc = top5.pct_ret.mean()
    rule = (f"<b>규칙 점검</b> · 최종 규칙 상위5: {', '.join(top5.name)} → 평균 {fmt_pct(tr,1)} vs 후보 평균 {fmt_pct(pr,1)}, "
            f"사후 백분위 평균 {pc*100:.0f} → {'적중' if tr > pr else '빗나감'}.")
    sgn = {"r1m": -1, "amt_ratio": -1, "vol20": -1, "h52": -1, "eps_g12": 1, "eps_g3": 1}
    agree = [f for f in d.index if f in sgn and np.sign(d[f]) == sgn[f] and abs(d[f]) >= 0.15]
    against = [f for f in d.index if f in sgn and np.sign(d[f]) != sgn[f] and abs(d[f]) >= 0.15]
    L = []
    if agree: L.append("일반 규칙과 같은 방향으로 갈림(" + ", ".join(LAB[f] for f in agree) + ")")
    if against: L.append("규칙과 반대로 갈림(" + ", ".join(LAB[f] for f in against) + f") — {mk}·{rate} 국면에서 추세·과열 종목이 이긴 예외 달")
    if not agree and not against: L.append("가격·실적 지표로는 갈리지 않은 달 — 업종 테마·개별 이벤트가 성과를 가름(정성 확인 영역)")
    ext = tb[tb.ret <= -0.30]
    if len(ext): L.append("급락 " + ", ".join(f"{r['name']} {r.ret*100:+.0f}%" for _, r in ext.iterrows()) + " — 개별 악재·기업행동(감자·분배·주식교환)·데이터 왜곡 여부를 공시로 확인해야 하는 유형")
    frz = g[g.zero_vol_days > 0]
    if len(frz): L.append("결정일 전 20일 내 거래 없는 날이 있던 종목(" + ", ".join(frz.name) + ") — 매수 전 제외")
    if m.kospi_vs_ma10 > 0.15: L.append("지수 과열권 — 후보 전체의 하방 위험이 커지므로 보유 수는 하한(5)으로")
    les = "<b>교훈</b> · " + "; ".join(L) + "."
    return [macro, val, mood, grow, res, disc, rule, les], (tr > pr)


def dtab(df, widths, fs=6.8, rh=4.3):
    t = Table([list(df.columns)] + df.astype(str).values.tolist(), colWidths=widths, rowHeights=rh*mm)
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "NG", fs), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e9c")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")), ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    return t


T = ic_table(F, FEATS); ic_chart(T, f"{OUT}/ic.png"); oos_chart(f"{OUT}/oos.png")
e = R["표본외(전반 선택→후반)"]; fa = R["최종규칙-전체"]
NARR = {t: narrative(t) for t in book}; hits = np.mean([v[1] for v in NARR.values()])
doc = SimpleDocTemplate(f"{OUT}/52wh_final_report.pdf", pagesize=landscape(A4), leftMargin=10*mm, rightMargin=10*mm, topMargin=9*mm, bottomMargin=8*mm,
                        title="52WH 후보 20개 — 6관점 사례분석과 선택 규칙 최종 리포트", author="regime-data")
S = [Paragraph("52WH 후보 20개 — 6관점 사례분석(74개월)과 선택 규칙 · 최종 리포트", H1),
     Paragraph("작성 2026-09-26 · 대상: 게이트 ON 결정월 2016-03 ~ 2026-07 (74개월 × 20종목 = 1,480건) · 체결 A안(결정월 말 다음 거래일 종가 매수 → 다음 월말 다음 거래일 종가 매도, 비용 미반영)", BD),
     Spacer(1, 4), Paragraph("최종 결론", H2),
     Paragraph("<b>1. 무엇을 살까 — '실적이 올라가는데 아직 과열되지 않은' 후보.</b> 20개 중 최근 공표 EPS가 상향된 종목(3개월 전 대비, 전년 대비)을 우선하고, "
               "거래대금이 평소의 몇 배로 급증하며 한 달 새 급등해 52주 고점에 정확히 닿은 종목은 뒤로 미룬다. 사전정보 중 월내 순위를 가장 일관되게 맞힌 것은 "
               f"EPS 3개월 변화(IC {T.loc['eps_g3','IC']:+.3f}, t {T.loc['eps_g3','t']:.1f}, 전·후반 모두 양)이고, 52주 고점 정확 도달(IC {T.loc['h52','IC']:+.3f}, t {T.loc['h52','t']:.1f})과 "
               "거래대금 급증·1개월 급등·단기 변동성(각 IC 약 −0.05, 후반 −0.06~−0.09)은 일관되게 불리했다.", BD),
     Paragraph("<b>2. 반드시 뺄 것 — 가격이 '딜 조건'에 묶인 종목과 거래정지 이력 종목.</b> 주식교환·공개매수·자산매각/청산이 진행 중인 종목은 52주 고점이어도 상승 여력이 막혀 있거나, "
               "기업행동이 수익을 왜곡한다(2026-07 동양생명·코람코더원리츠). 가격 데이터로는 안 보이고 공시·뉴스로만 보이는 정성 영역이다.", BD),
     Paragraph("<b>3. 판별력이 없었던 것 — 밸류에이션·적자 여부·업종·거시.</b> PER·PBR·배당(IC ±0.04 이내), 적자 여부, 업종군, 금리·신용·환율·나스닥 국면별 업종 차이는 "
               "통계적으로 구분되지 않았다. 다만 적자·과열 종목은 평균은 높고 중앙값은 낮은 '복권형' 분포(월내 하위 25%에 빠진 비율 28~35% vs 실적 상향 종목 22%)라, "
               "5개로 좁힐 때는 피하는 편이 결과의 흔들림을 줄인다.", BD),
     Paragraph(f"<b>4. 기대 효과는 작다 — 정성 판단이 나머지를 채워야 한다.</b> 전반(2016–20)에서만 고른 규칙을 후반(2021–26)에 적용한 표본외 결과: IC {e.IC:+.3f}(t {e.IC_t:.1f}), "
               f"상위5 연환산 {e.top5_연환산*100:.0f}% vs 후보 20개 {e.후보20_연환산*100:.0f}% vs 같은 달 KOSPI {e.KOSPI_연환산*100:.0f}%, 상위5가 후보 평균을 이긴 달 {e.top5이_후보평균_상회*100:.0f}%. "
               f"전체표본으로 정의한 최종 규칙은 IC {fa.IC:+.3f}(표본내라 과대평가), 74개월 중 상위5가 후보 평균을 이긴 달 {hits*100:.0f}%. "
               "A안에서 KOSPI를 확실히 이기려면 IC 0.15~0.2가 필요하므로(8차 스펙트럼) 기계 규칙만으로는 부족하다. 정성 평가가 IC를 0.05~0.1 더 올려야 하며, "
               "그 추가분은 데이터가 못 보는 곳 — 실적 발표의 질·컨센서스 변화, 딜·이벤트, 수주·신제품 같은 촉매 — 에서 나와야 한다.", BD),
     Paragraph("<b>5. 운용 절차.</b> ① 매월 20개를 규칙 순위(월별 표의 '규칙순위')로 1차 정렬 → ② 딜·정지·분배 공시 확인 후 제외 → ③ 상위권부터 정성 채점(실적 상향의 질, 촉매, 과열 여부) → "
               "④ 5개 매수, 적합 종목이 5개 미만이면 나머지는 현금 → ⑤ 채점을 매수 전 커밋해 forward IC로 판단력을 측정.", BD),
     Paragraph("<b>한계</b> · 사후 수익을 알고 쓰는 사례분석은 사후합리화 위험이 크다. 이를 막기 위해 판별 요인은 모두 결정일 정보로만 계산했고, 결론은 74개월 통계와 표본외 검증에만 근거했다. "
               "EPS는 KRX 월말 공표값(연간 결산 기준으로 보이며 분기 갱신 여부는 확인 필요). 컨센서스·기관 수급·DART 재무는 이번 버전에 미반영(DART 수집 시간초과, 재설계 예정). "
               "뉴스 원문 조사는 2026-07 결정월만 수행했고 나머지 달의 논리는 데이터 기반 추론이다. 코람코더원리츠처럼 분배·교환이 수익률에 반영되지 않은 사례가 있어 개별 수치는 과장될 수 있다.", BD), PageBreak()]
mt = [["관점", "사용한 사전정보(결정일 기준)", "출처"],
      ["거시경제", "KOSPI 1/3/12개월, 10개월선 이격, CD·국고3Y 3개월 변화, 신용스프레드(BBB−−AA−), 원/달러 3개월, 미10Y 3개월, 나스닥 3개월", "KRX, ECOS(1개월 지연), FRED"],
      ["기업가치평가", "PER(적자면 제외), PBR, E/P, B/P, 배당수익률", "KRX 월말 공표 PER·PBR·EPS·BPS·DIV"],
      ["재무", "적자(EPS≤0), 자본잠식(BPS≤0), 거래 없는 날", "KRX, 가격 패널"],
      ["시장 분위기", "1/3/6/12개월 상승률, 52주 고점 근접도·체류일수, 20일 신고가 횟수, 변동성, 60일 낙폭, 회전율·거래대금 급증도(20/120일), 업종 쏠림", "가격·거래량·거래대금 패널"],
      ["기대치", "최근 3개월 EPS 변화(실적 상향 반영), PER 수준(기대 선반영), 1개월 급등(기대 과열) — 컨센서스는 미확보", "KRX"],
      ["성장성", "EPS 전년비, BPS 전년비", "KRX"]]
tt = Table([[Paragraph(c, SM) for c in r] for r in mt], colWidths=[28*mm, 190*mm, 59*mm])
tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c6dbef")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
S += [Paragraph("방법 — 6관점을 사전정보 피처로 옮기기", H1), tt, Spacer(1, 4),
      Paragraph("<b>채점</b> · 각 월 20개 안에서 피처 순위와 사후 수익 순위의 상관(월내 IC)을 구해 74개월 평균·t값을 본다. 시장 전체 등락은 월 안에서 상쇄되므로 '20개 중 무엇을 고를까'만 측정한다. "
                "<b>표본외 검증</b> · 2016–20 자료만으로 |t|≥1.5 피처를 골라 부호대로 합성 → 2021–26에 그대로 적용. <b>최종 규칙</b> · EPS 3개월 변화(+), EPS 전년비(+), 거래대금 급증도(−), "
                "1개월 상승률(−), 단기 변동성(−), 52주 고점 근접도(−)의 월내 백분위 합(동점은 티커순으로 처리해 사후 순서 누설 차단).", BD),
      Paragraph("피처별 월내 IC (74개월)", H2), Image(f"{OUT}/ic.png", width=250*mm, height=250*mm*4.4/11.2), PageBreak()]
tab = T[["IC", "t", "hit", "IC_16_20", "IC_21_26", "n"]].sort_values("t").round(3).reset_index().rename(columns={"index": "피처", "hit": "양(+)인 달 비율", "IC_16_20": "IC 2016–20", "IC_21_26": "IC 2021–26", "n": "월수"})
tab.insert(1, "관점", tab["피처"].map(PERSP))
S += [Paragraph("피처별 IC 상세 · 전반/후반 일관성", H2), dtab(tab, [28*mm, 32*mm] + [22*mm]*6), PageBreak()]
Rt = R.copy()
for c in Rt.columns:
    Rt[c] = [f"{v:.0f}" if i == "월수" else (f"{v*100:.1f}%" if ("월평균" in i or "연환산" in i or "상회" in i) else f"{v:.3f}") for i, v in Rt[c].items()]
Rt = Rt.reset_index().rename(columns={"index": "지표"})
S += [Paragraph("표본외 검증 — 규칙이 처음 보는 기간에서도 통했나", H1), dtab(Rt, [40*mm] + [48*mm]*4, 7, 4.6), Spacer(1, 3),
      Image(f"{OUT}/oos.png", width=250*mm, height=250*mm*3.3/11.2),
      Paragraph("해석 · 순위 상관(IC)은 양이지만 상위5의 초과수익은 작다. 후보 20개 수익이 소수 대박 종목에 좌우되는 비대칭 분포라, 순위를 조금 맞혀도 평균 수익 차이로는 잘 드러나지 않는다. "
                "'표본내' 수치는 전체 기간을 보고 정한 규칙이라 과대평가돼 있다.", BD), PageBreak()]
S += [Paragraph("연도별 주요 사건 (일반 지식 요약 — 날짜·수치는 원문 확인 권장)", H1)] + [Paragraph(f"<b>{y}</b> · {txt}", BD) for y, txt in TIMELINE] + [PageBreak()]
t0 = max(book); tb0 = book[t0]["table"]
rows = [["순위", "종목", "보유수익", "당시 상황 / 뉴스 요약 (출처)"]] + [[str(r["rank"]), r["name"], f"{r.ret*100:+.1f}%", Paragraph(NEWS.get(r["name"], "조사 미완료"), SM)] for _, r in tb0.iterrows()]
T4 = Table(rows, colWidths=[10*mm, 28*mm, 17*mm, 222*mm], repeatRows=1)
T4.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "NG", 7), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e9c")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc"))]))
S += [Paragraph(f"{t0} 결정월 — 뉴스 조사", H1), Paragraph(THEME, BD), T4, PageBreak()]
S += [Paragraph("월별 사례 (최신 → 과거) · 표의 '규칙순위' 노란 칸 = 최종 규칙 상위 5", H1)]
for t in sorted(book, reverse=True):
    b = book[t]; month_chart(b, f"{OUT}/m_{t}.png"); tbm = b["table"]
    S += [CondPageBreak(125*mm), Paragraph(f"결정 {t} · 매수 {b['buy'].date()} → 매도 {b['sell'].date()} · 후보 평균 {tbm.ret.mean()*100:+.1f}% · KOSPI {b['kospi'].iloc[-1]*100:+.1f}% · 최선 {tbm.ret.max()*100:+.1f}% / 최악 {tbm.ret.min()*100:+.1f}%", H2),
          Image(f"{OUT}/m_{t}.png", width=277*mm, height=277*mm*2.6/11.2), month_table(t), Spacer(1, 2)]
    S += [Paragraph(p, NR) for p in NARR[t][0]] + [Spacer(1, 6)]
doc.build(S)
print("ok hits", round(hits, 3))
