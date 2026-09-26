"""월별 52WH 후보 20개 사후 성과북 PDF (A안 체결: 월말+1거래일 종가 매수 → 다음 월말+1거래일 종가)."""
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from news import NEWS, THEME

FP = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"; FB = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
fm.fontManager.addfont(FP); plt.rcParams["font.family"] = fm.FontProperties(fname=FP).get_name(); plt.rcParams["axes.unicode_minus"] = False
pdfmetrics.registerFont(TTFont("NG", FP)); pdfmetrics.registerFont(TTFont("NGB", FB))
H1 = ParagraphStyle("h1", fontName="NGB", fontSize=15, leading=19, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="NGB", fontSize=11, leading=14, spaceAfter=3)
BD = ParagraphStyle("bd", fontName="NG", fontSize=8.5, leading=12)
SM = ParagraphStyle("sm", fontName="NG", fontSize=6.6, leading=8.2)
book = pickle.load(open("monthbook.pkl", "rb"))
cmap = plt.get_cmap("RdYlGn_r")
OUT = "/home/claude/book"

def month_chart(t, b, path):
    fig, ax = plt.subplots(figsize=(11.2, 3.3), dpi=150)
    P = b["path"]; n = P.shape[1]
    for i, c in enumerate(P.columns):
        ax.plot(P.index, P[c] * 100, color=cmap(i / (n - 1)), lw=1.1 if i in (0, n - 1) else 0.7, alpha=0.95)
    ax.plot(P.index, P.mean(axis=1) * 100, color="#1f4e9c", lw=2.2, label="후보 20개 평균")
    ax.plot(b["kospi"].index, b["kospi"] * 100, color="black", lw=1.6, ls="--", label="KOSPI")
    tb = b["table"]
    for i in (0, n - 1):
        ax.annotate(f"{i+1}. {tb.name.iloc[i]}", (P.index[-1], P.iloc[-1, i] * 100), fontsize=7, xytext=(4, 0), textcoords="offset points", va="center")
    ax.axhline(0, color="grey", lw=0.6); ax.set_ylabel("매수 후 누적수익 (%)"); ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=7, frameon=False); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def cover_chart(path):
    rows = []
    for t, b in book.items():
        r = b["table"].ret; rows.append((t.to_timestamp(), r.max(), r.median(), r.min(), r.mean(), b["kospi"].iloc[-1]))
    D = pd.DataFrame(rows, columns=["t", "best", "med", "worst", "mean", "kospi"]).set_index("t")
    fig, ax = plt.subplots(figsize=(11.2, 3.6), dpi=150)
    ax.vlines(D.index, D.worst.clip(lower=-0.6) * 100, D.best.clip(upper=1.2) * 100, color="#9ecae1", lw=3, label="최악~최선 범위(게이트 ON 월)")
    ax.scatter(D.index, D["mean"] * 100, s=9, color="#1f4e9c", zorder=3, label="후보 20개 평균")
    ax.scatter(D.index, D.kospi * 100, s=9, marker="x", color="black", zorder=3, label="KOSPI")
    ax.set_ylim(-60, 120); ax.axhline(0, color="grey", lw=0.6); ax.grid(alpha=0.25); ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_ylabel("보유 1개월 수익 (%)"); ax.tick_params(labelsize=7); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return D

def trunc(s, n):
    s = "" if not isinstance(s, str) else s.strip()
    return s if len(s) <= n else s[:n - 1] + "…"

def month_table(tb):
    head = ["순위", "종목", "티커", "KRX 업종", "한 줄 설명 (주요제품 / 표준산업분류)", "보유수익", "기간 MDD"]
    data = [head]
    for _, r in tb.iterrows():
        desc = trunc(r.products, 34) or "–"
        ind = trunc(r.industry, 26)
        line = f"{desc} / {ind}" if ind else desc
        if r.delisted: line = "[상폐] " + line
        data.append([str(r["rank"]), trunc(r["name"], 12), r.ticker, trunc(r.sector, 9), Paragraph(line, SM), f"{r.ret*100:+.1f}%", f"{r.mdd*100:.1f}%"])
    T = Table(data, colWidths=[9*mm, 26*mm, 15*mm, 21*mm, 160*mm, 17*mm, 17*mm], rowHeights=[5.2*mm] + [4.35*mm]*len(tb), repeatRows=1)
    st = [("FONT", (0, 0), (-1, 0), "NGB", 7), ("FONT", (0, 1), (-1, -1), "NG", 6.8),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e9c")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("ALIGN", (0, 0), (0, -1), "CENTER"), ("ALIGN", (5, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")), ("TOPPADDING", (0, 0), (-1, -1), 0.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6)]
    for i, (_, r) in enumerate(tb.iterrows(), 1):
        c = cmap((i - 1) / (len(tb) - 1)); st.append(("TEXTCOLOR", (5, i), (5, i), colors.Color(c[0]*0.8, c[1]*0.8, c[2]*0.8)))
    T.setStyle(TableStyle(st)); return T

doc = SimpleDocTemplate(f"{OUT}/52wh_monthbook.pdf", pagesize=landscape(A4), leftMargin=10*mm, rightMargin=10*mm, topMargin=9*mm, bottomMargin=8*mm,
                        title="52WH 후보 20개 월별 사후 성과북", author="regime-data")
S = []
D = cover_chart(f"{OUT}/cover.png")
S += [Paragraph("52WH 후보 20개 — 월별 사후 성과북 (2016-03 ~ 2026-07 결정월)", H1),
      Paragraph("목적: 매월 기계가 뽑은 후보 20개가 실제로 어떻게 갈렸는지(최선~최악) 보고, 정성 재선별로 무엇을 거를 수 있었는지 감을 잡기 위한 자료. "
                "각 월 1쪽: 20개 누적수익 경로 대조 차트 + 사후 수익 순위표.", BD), Spacer(1, 3),
      Paragraph("<b>정의</b> · 후보: 시총 상위 500 보통주 중 수정가/252일 최고가 상위 20 (기준형과 동일). 게이트(KOSPI 월말 > 10개월 평균)가 켜진 달만 수록(74개월). "
                "체결: 결정월 말 다음 거래일 종가 매수 → 다음 월말 다음 거래일 종가 매도(A안, 비용 미반영). 순위: 보유기간 수익 내림차순. 기간 MDD: 보유기간 중 고점 대비 최대 낙폭.", BD),
      Paragraph("<b>주의</b> · 한 줄 설명은 2026-09 기준 KRX 상장정보(주요제품·표준산업분류)라 과거 시점과 다를 수 있음. 상폐 종목은 상폐 목록의 산업분류만 표시. "
                "KRX 업종은 결정 시점 스냅샷. 수정주가는 주식수 역비율 매칭으로 복원 — 특별분배·주식교환 등은 반영되지 않아 수익이 왜곡될 수 있음(2026-07 코람코더원리츠·동양생명 사례).", BD),
      Spacer(1, 4), Image(f"{OUT}/cover.png", width=270*mm, height=270*mm*3.6/11.2)]
sp = D.best - D.worst; top = (D["mean"] > D.kospi).mean()
S += [Paragraph(f"요약 · 월평균 최선-최악 격차 {sp.mean()*100:.1f}%p (중앙값 {sp.median()*100:.1f}%p) · 후보 평균이 KOSPI를 이긴 달 {top*100:.0f}% · "
                f"최선 종목 평균 {D.best.mean()*100:+.1f}% · 최악 종목 평균 {D.worst.mean()*100:+.1f}%", BD), PageBreak()]
# 2026-07 뉴스 섹션
t0 = max(book)
S += [Paragraph(f"{t0} 결정월 — 종목별 당시 상황 (뉴스 조사)", H1), Paragraph(THEME, BD), Spacer(1, 4)]
tb = book[t0]["table"]
rows = [["순위", "종목", "보유수익", "당시 상황 / 뉴스 요약 (출처)"]]
for _, r in tb.iterrows():
    rows.append([str(r["rank"]), r["name"], f"{r.ret*100:+.1f}%", Paragraph(NEWS.get(r["name"], "조사 미완료 — 다음 회차 보강"), SM)])
T = Table(rows, colWidths=[10*mm, 28*mm, 17*mm, 222*mm], repeatRows=1)
T.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "NGB", 7.5), ("FONT", (0, 1), (-1, -1), "NG", 7),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e9c")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                       ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc"))]))
S += [T, PageBreak()]
for t in sorted(book, reverse=True):
    b = book[t]; month_chart(t, b, f"{OUT}/m_{t}.png")
    tbm = b["table"]; km = b["kospi"].iloc[-1]
    S += [KeepTogether([Paragraph(f"결정 {t} · 매수 {b['buy'].date()} → 매도 {b['sell'].date()} · 후보 평균 {tbm.ret.mean()*100:+.1f}% · KOSPI {km*100:+.1f}% · "
                                  f"최선 {tbm.ret.max()*100:+.1f}% / 최악 {tbm.ret.min()*100:+.1f}%", H2),
                        Image(f"{OUT}/m_{t}.png", width=277*mm, height=277*mm*3.3/11.2), month_table(tbm)]), PageBreak()]
doc.build(S)
print("ok")
