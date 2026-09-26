"""토론 산출 scores.csv 검증 → 사전등록 규칙(70점 이상 점수순 최대 5, 0개면 현금)으로 선정 → 텔레그램 문안.
왜 코드로: 선정을 LLM에 맡기면 개수 맞추기식 점수 조정이 섞일 수 있음."""
import sys
from pathlib import Path
import pandas as pd

SCORE_TH, MAX_PICK = 70, 5          # forward/run_forward.py 와 동일 (사전등록)
IN, D = Path(sys.argv[1]), sys.argv[2]
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
sc = sc.sort_values("score", ascending=False)
pick = sc[sc.score >= SCORE_TH].head(MAX_PICK)
sc["selected"] = sc.ticker.isin(pick.ticker).astype(int)
sc.to_csv(IN / f"{D}_scores.csv", index=False)
L = [f"📊 {D} ③v2 결과 (기준 {SCORE_TH}점, 최대 {MAX_PICK}개)"]
L += [f"✅ {r.name} {r.score:.0f}점 — {r.reason}" for r in pick.itertuples()] or ["선정 0개 → 전액 현금"]
L += ["", "미선정 상위: " + ", ".join(f"{r.name} {r.score:.0f}" for r in sc[sc.selected == 0].head(5).itertuples()),
      "전문: 리포 " + str(IN / f"{D}_debate.md")]
(IN / f"{D}_telegram.txt").write_text("\n".join(L))
print("\n".join(L))
