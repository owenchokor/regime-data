"""축 세션 직후: 실행기록 보존(메트릭용, 커밋 안 함) + 산출 파일이 후보 전 종목을 담았는지 확인.
왜: 불완전한 축 의견으로 종합까지 가면 사용량만 쓰고 실패(20종목 1차 측정에서 G·V 헤더만 남음)."""
import os, shutil, sys
from pathlib import Path
import pandas as pd

k, ex = sys.argv[1], sys.argv[2]
IN, D = Path(os.environ["IN"]), os.environ["D"]
if ex and Path(ex).exists():
    shutil.copy(ex, Path(os.environ["RUNNER_TEMP"]) / f"exec_{k}.json")
tk = pd.read_csv(IN / f"{D}.csv", dtype={"ticker": str}).ticker
f = IN / "work" / f"{D}_{k}.md"
txt = f.read_text() if f.exists() else ""
miss = [t for t in tk if f"### {t}" not in txt]
if miss:
    print(f"::error::{k}축 산출 불완전: {f.name} 누락 {len(miss)}/{len(tk)} {miss[:5]} / work 폴더 {sorted(p.name for p in (IN / 'work').glob('*'))}")
    sys.exit(1)
print(k, "ok", len(tk))
