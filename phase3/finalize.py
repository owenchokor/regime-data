"""토론 산출 scores.csv 검증 → 사전등록 규칙으로 선정 → 텔레그램 문안(구조화) 생성.
규칙: 70점 이상 점수순 최대 5, 0개면 현금. 게이트 OFF(KOSPI<MA10)면 매수 없이 채점만 기록(shadow).
왜 코드로: 선정을 LLM에 맡기면 개수 맞추기식 점수 조정이 섞일 수 있음."""
import json, sys
from pathlib import Path
import pandas as pd

SCORE_TH, MAX_PICK = 70, 5          # forward/run_forward.py 와 동일 (사전등록)
AVOID_TH = 30                       # 루브릭 0–29 '명확한 회피' 구간 경계
IN, D = Path(sys.argv[1]), sys.argv[2]
P = json.loads((Path(__file__).parent / "personas.json").read_text())
cand = pd.read_csv(IN / f"{D}.csv", dtype={"ticker": str})
f = IN / f"{D}_scores.csv"
if not f.exists():
    print(f"::error::scores 파일 없음 {f} / 폴더 내용 {sorted(p.name for p in IN.rglob('*'))}"); sys.exit(1)
try:
    sc = pd.read_csv(f, dtype={"ticker": str})
except Exception as e:
    print(f"::error::scores 파싱 실패 {e}"); sys.exit(1)
sc["ticker"] = sc.ticker.str.zfill(6)
err = []
if set(sc.ticker) != set(cand.ticker): err.append(f"종목 불일치: 누락 {set(cand.ticker) - set(sc.ticker)}, 초과 {set(sc.ticker) - set(cand.ticker)}")
if sc.ticker.duplicated().any(): err.append("중복 티커")
if not sc.score.between(0, 100).all(): err.append("점수 범위 밖")
if err:
    print("::error::검증 실패: " + "; ".join(err)); sys.exit(1)   # 왜 ::error::: annotation으로 남아 로그 없이도 원인 확인

sc = sc.drop(columns=[c for c in ("selected", "shadow", "rule_rank") if c in sc]).merge(
    cand[["ticker", "rule_rank", "sector"]], on="ticker", how="left").sort_values(["score", "rule_rank"], ascending=[False, True])
gate = int(cand.gate.iloc[0]) if "gate" in cand and pd.notna(cand.gate.iloc[0]) else 1
pick = sc[sc.score >= SCORE_TH].head(MAX_PICK)
sc["shadow"] = sc.ticker.isin(pick.ticker).astype(int)          # 규칙상 선정(게이트 무관)
sc["selected"] = sc.shadow * gate                                # 실제 편입
sc.to_csv(f, index=False)

def axes(r):
    return f"성장 {r.g:.0f} · 밸류 {r.v:.0f} · 시장 {r.m:.0f}"

n_ex = int((cand.get("deal_flag", pd.Series(dtype=str)).isin(["1", 1, True])).sum()) if "deal_flag" in cand else 0
pend = "pending" in set(cand.get("deal_flag", pd.Series(dtype=str)).astype(str))
L = [f"📊 {D} ③v2 결정  (기준일 {cand.as_of.iloc[0]} · 게이트 {'ON' if gate else 'OFF'})", "",
     "【선정 과정】",
     "① 유니버스: 보통주·상장 13개월↑·시총 상위 500",
     f"② 후보: 52주 고가 근접 상위 {len(cand)}종목",
     f"③ 기계 배제: 딜·재무 플래그 {'대기(DART 필터 사전등록 전)' if pend else f'{n_ex}종목'}",
     f"④ 채점: {P['G']['name']}·{P['V']['name']}·{P['M']['name']} 독립 의견 → {P['S']['name']} 반박·종합",
     f"⑤ 편입: {SCORE_TH}점↑ 점수순 최대 {MAX_PICK}개 → {len(pick)}종목" + ("" if gate else " (게이트 OFF → 매수 없음, 기록만)"), ""]
if len(pick):
    L.append("✅ 편입" if gate else "📝 규칙상 선정 (게이트 OFF — 매수 안 함)")
    L += [f"{i}. {r.name} {r.score:.0f}점  [{axes(r)}]\n   {r.reason}" for i, r in enumerate(pick.itertuples(), 1)]
else:
    L.append("선정 0개 → 전액 현금")
rest = sc[~sc.ticker.isin(pick.ticker)]
watch, avoid = rest[rest.score >= AVOID_TH], rest[rest.score < AVOID_TH]
L += ["", f"⏸ 관망 {len(watch)}: " + ", ".join(f"{r.name} {r.score:.0f}" for r in watch.itertuples()),
      f"⛔ 회피 {len(avoid)}: " + ", ".join(f"{r.name} {r.score:.0f}" for r in avoid.itertuples()), "",
      f"채점표 이미지 별도 전송 · 토론 전문: https://github.com/owenchokor/regime-data/blob/main/{IN}/{D}_debate.md"]
(IN / f"{D}_telegram.txt").write_text("\n".join(L))
print("\n".join(L))
