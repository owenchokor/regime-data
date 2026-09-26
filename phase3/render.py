"""③v2 채점표 PNG: 선정 퍼널 + 종목별(최종순위·52WH 규칙순위 변화·3축 히트셀·최종점수 막대·상태).
왜 이미지: 텔레그램 줄글만으로는 20종목의 순위·축별 차이·기준선 대비 위치가 한눈에 안 보임."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Polygon
import pandas as pd

SCORE_TH, AVOID_TH = 70, 30          # finalize.py 와 동일
BG, PANEL, GRID, TEXT, DIM = "#0b0f17", "#121826", "#1f2937", "#e5e7eb", "#8b95a7"
CYAN, MAG, LIME, AMBER, RED = "#22d3ee", "#e879f9", "#a3e635", "#fbbf24", "#f87171"
HEAT = LinearSegmentedColormap.from_list("h", ["#3b0d1a", "#7f1d1d", "#a16207", "#3f6212", "#15803d", "#22c55e"])

for fp in ("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, fc, ec=None, r=0.012, lw=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or fc, lw=lw, zorder=z))


def render(IN: Path, D: str) -> Path:
    P = json.loads((Path(__file__).parent / "personas.json").read_text())
    cand = pd.read_csv(IN / f"{D}.csv", dtype={"ticker": str})
    sc = pd.read_csv(IN / f"{D}_scores.csv", dtype={"ticker": str}).sort_values(["score", "rule_rank"], ascending=[False, True]).reset_index(drop=True)
    gate = int(cand.gate.iloc[0]) if pd.notna(cand.gate.iloc[0]) else 1
    n = len(sc); n_pick = int(sc.shadow.sum())

    rowh = 0.034; top = 0.80; H = top - n * rowh
    fig = plt.figure(figsize=(12, 3.2 + n * 0.42), dpi=170, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(H - 0.05, 1); ax.axis("off")

    # 제목
    ax.text(0.03, 0.965, f"③v2 채점표  ·  {D}", color=TEXT, fontsize=22, va="top", weight="bold")
    ax.text(0.03, 0.925, f"기준일 {cand.as_of.iloc[0]}   ·   게이트 {'ON' if gate else 'OFF (매수 없음, 기록만)'}   ·   "
            f"편입 기준 {SCORE_TH}점↑ 최대 5", color=DIM, fontsize=11, va="top")
    ax.text(0.97, 0.965, f"{n_pick if gate else 0}", color=LIME if gate and n_pick else AMBER, fontsize=34, ha="right", va="top", weight="bold")
    ax.text(0.97, 0.905, "편입 종목", color=DIM, fontsize=10, ha="right", va="top")

    # 퍼널
    steps = [("유니버스", "시총 상위 500", CYAN), ("후보", f"52WH 상위 {n}", CYAN),
             ("기계 배제", "딜·재무 대기", DIM), ("3축 채점", "독립→반박→종합", MAG),
             (f"{SCORE_TH}점↑", f"{int((sc.score >= SCORE_TH).sum())}종목", AMBER), ("편입", f"{n_pick if gate else 0}종목", LIME)]
    fx, fw, fy, fh = 0.03, 0.94 / len(steps), 0.835, 0.052
    for i, (a, b, c) in enumerate(steps):
        x0 = fx + i * fw
        pts = [(x0, fy), (x0 + fw - 0.012, fy), (x0 + fw, fy + fh / 2), (x0 + fw - 0.012, fy + fh), (x0, fy + fh), (x0 + (0.012 if i else 0), fy + fh / 2)]
        ax.add_patch(Polygon(pts, closed=True, fc=PANEL, ec=c, lw=1.4))
        ax.text(x0 + fw / 2, fy + fh * 0.66, a, color=c, fontsize=10.5, ha="center", va="center", weight="bold")
        ax.text(x0 + fw / 2, fy + fh * 0.28, b, color=TEXT, fontsize=8.5, ha="center", va="center")

    # 헤더
    cols = dict(rank=0.045, name=0.075, rule=0.335, g=0.425, v=0.49, m=0.555, bar0=0.615, bar1=0.87, st=0.935)
    hy = top + 0.008
    for k, t in [("rank", "최종"), ("name", "종목 · 업종"), ("rule", "52WH순위→최종"),
                 ("g", P["G"]["name"]), ("v", P["V"]["name"]), ("m", P["M"]["name"]), ("st", "상태")]:
        ax.text(cols[k] + (0.0325 if k in "gvm" else 0), hy, t, color=DIM, fontsize=8.5, ha="center" if k in ("rank", "g", "v", "m", "st", "rule") else "left", va="bottom")
    ax.text(cols["bar0"], hy, "최종 점수", color=DIM, fontsize=8.5, va="bottom")

    xs = lambda v: cols["bar0"] + (cols["bar1"] - cols["bar0"]) * v / 100
    for i, r in sc.iterrows():
        y = top - (i + 1) * rowh; yc = y + rowh / 2
        pick = bool(r.shadow)
        status, sc_c = (("편입" if gate else "선정*"), LIME) if pick else (("관망", AMBER) if r.score >= AVOID_TH else ("회피", RED))
        box(ax, 0.03, y + 0.003, 0.94, rowh - 0.006, PANEL if not pick else "#132a1b", ec=LIME if pick else PANEL, lw=1.0 if pick else 0)
        ax.text(cols["rank"], yc, f"{i + 1}", color=LIME if pick else TEXT, fontsize=12, ha="center", va="center", weight="bold")
        ax.text(cols["name"], yc + 0.004, r["name"], color=TEXT, fontsize=11, va="center", weight="bold")
        ax.text(cols["name"], yc - 0.0085, f"{r.ticker} · {r.sector}", color=DIM, fontsize=7.5, va="center")
        d = int(r.rule_rank) - (i + 1)
        arrow, ac = ("▲", LIME) if d > 0 else (("▼", RED) if d < 0 else ("＝", DIM))
        ax.text(cols["rule"], yc, f"{int(r.rule_rank):>2} → {i + 1:<2} {arrow}{abs(d) if d else ''}", color=ac, fontsize=9.5, ha="center", va="center")
        for k in ("g", "v", "m"):
            v = float(r[k]); box(ax, cols[k], y + 0.006, 0.065, rowh - 0.012, HEAT(v / 100), r=0.006, z=2)
            ax.text(cols[k] + 0.0325, yc, f"{v:.0f}", color="white", fontsize=10, ha="center", va="center", weight="bold", zorder=3)
        box(ax, cols["bar0"], yc - 0.006, cols["bar1"] - cols["bar0"], 0.012, GRID, r=0.004, z=2)
        box(ax, cols["bar0"], yc - 0.006, xs(r.score) - cols["bar0"], 0.012, sc_c, r=0.004, z=3)
        ax.text(xs(r.score) + 0.006, yc, f"{r.score:.0f}", color=sc_c, fontsize=10.5, va="center", weight="bold", zorder=4)
        ax.text(cols["st"], yc, status, color=sc_c, fontsize=10, ha="center", va="center", weight="bold")
    ax.plot([xs(SCORE_TH)] * 2, [H, top], color=LIME, lw=1.2, ls=(0, (4, 3)), zorder=5)
    ax.text(xs(SCORE_TH), top + 0.004, f"기준 {SCORE_TH}", color=LIME, fontsize=8, ha="center", va="bottom")
    ax.plot([xs(AVOID_TH)] * 2, [H, top], color=RED, lw=0.8, ls=(0, (2, 4)), alpha=0.6, zorder=5)

    foot = f"히트셀: 각 필명의 축 점수(0~100). 최종 점수는 {P['S']['name']}가 반박 후 종합(단순 평균 아님). 52WH순위→최종: 기계 규칙 순위 대비 재선별 변화."
    if not gate:
        foot += "  *게이트 OFF: 규칙상 선정이나 매수 안 함."
    ax.text(0.03, H - 0.02, foot, color=DIM, fontsize=8, va="top")
    out = IN / f"{D}_scorecard.png"
    fig.savefig(out, facecolor=BG); plt.close(fig)
    return out


if __name__ == "__main__":
    print(render(Path(sys.argv[1]), sys.argv[2]))
