"""Baseline + walk-forward 실행 → results/*.pkl."""
import pickle
import numpy as np
import pandas as pd
from core import *
from strategies import *

d = load_daily()
m = to_monthly(d)
f = features(m)
r = m["kospi"].pct_change()
rq = m["kosdaq"].pct_change()
ONE = pd.Series(1.0, index=r.index)

res: dict = {"m": m, "f": f, "r": r, "rq": rq}

# ── Baseline (사전 고정 파라미터; MA10 = Faber(2007) 관례) ──
base_pos = {
    "KOSPI B&H": ONE,
    "TSMOM1 (월간 모멘텀)": (f["mom1"] > 0).astype(float),
    "TSMOM12 (12M 모멘텀)": (f["mom12"] > 0).astype(float),
    "PxMA10 (이평 타이밍)": (f["pma10"] > 0).astype(float),
}
res["base_pos"] = base_pos

# ── 룰 그리드 (고정 파라미터 전부 — SPA/스누핑 카운트용) ──
fam = rule_grid(f)
res["fam"] = fam

# ── Nested 룰 선택 ──
nested = {}
for window in ["expanding", "rolling"]:
    for fname, cands in fam.items():
        pos, picks = nested_rule(cands, r, window)
        nested[(fname, window)] = (pos, picks)
    pos, picks = nested_all_rules(fam, r, window)
    nested[("ALL", window)] = (pos, picks)
res["nested"] = nested
res["committee"] = committee(f)

# ── ML walk-forward ──
ml = {}
for window in ["expanding", "rolling"]:
    ml[window] = ml_walkforward(f, r, window)
res["ml"] = ml
pickle.dump(res, open("/home/claude/work/wf.pkl", "wb"))
print("done")
