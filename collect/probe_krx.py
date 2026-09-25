"""
KRX 호출 한도 진단 — get_market_cap_by_ticker 가 ~194콜 후 KeyError로 막히는 원인 규명
A: 무지연 연속 호출 → 첫 실패까지 몇 콜/몇 초
B: 강제 재로그인으로 복구되는가
C: 대기(30/60/120/300초)로 복구되는가
D: 콜 간 지연(1.0s)을 주면 벽을 피하는가
결과: data/probe_log.txt
"""
import os, time
from pathlib import Path
import pandas as pd
from pykrx import stock
from pykrx.website.comm import auth

OUT = Path("data"); OUT.mkdir(exist_ok=True)
LOG = open(OUT / "probe_log.txt", "w")
def log(*a):
    m = f"[{time.strftime('%H:%M:%S')}] " + " ".join(map(str, a))
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

days = [d.strftime("%Y%m%d") for d in pd.read_parquet(OUT / "kospi.parquet").index]

def one(ds: str, mkt: str = "KOSPI"):
    """(성공여부, 행수, 소요초, 에러요약)"""
    t = time.time()
    try:
        df = stock.get_market_cap_by_ticker(ds, market=mkt)
        return (not df.empty), len(df), time.time() - t, ""
    except Exception as e:
        return False, 0, time.time() - t, f"{type(e).__name__}: {str(e)[:80]}"

def relogin() -> bool:
    auth.set_auth_session(None)          # 왜: 전역 세션 폐기 → 다음 호출이 새 로그인
    s = auth.get_auth_session()
    ok = s is not None and s.is_valid()
    log("  relogin ->", ok)
    return ok

# ── A: 무지연 연속 ──────────────────────────────────────────────
log("=== A: 무지연 연속 호출 ===")
t0 = time.time(); n_ok = 0; wall_at = None
for i, ds in enumerate(days[:400]):
    ok, n, dt, err = one(ds)
    if ok:
        n_ok += 1
        if i % 25 == 0: log(f"  {i} {ds} n={n} {dt:.2f}s")
    else:
        wall_at = i
        log(f"  WALL at call#{i} ({n_ok} 성공, 경과 {time.time()-t0:.0f}s) {err}")
        break
if wall_at is None:
    log("  400콜 전부 성공 — 벽 없음"); LOG.close(); raise SystemExit(0)

# ── B: 재로그인 복구 ────────────────────────────────────────────
log("=== B: 재로그인 복구 ===")
relogin()
ok, n, dt, err = one(days[wall_at])
log(f"  재로그인 직후: ok={ok} n={n} {err}")

# ── C: 대기 복구 ────────────────────────────────────────────────
if not ok:
    log("=== C: 대기 복구 ===")
    for w in (30, 60, 120, 300):
        log(f"  sleep {w}s ..."); time.sleep(w)
        ok, n, dt, err = one(days[wall_at])
        log(f"  {w}s 후: ok={ok} n={n} {err}")
        if ok: break
    if not ok:
        log("  대기+재로그인 재시도"); relogin()
        ok, n, dt, err = one(days[wall_at])
        log(f"  최종: ok={ok} n={n} {err}")

# ── D: 지연 1.0s 로 60콜 ────────────────────────────────────────
log("=== D: 콜간 1.0s 지연, 60콜 ===")
t0 = time.time(); fail = 0
for i, ds in enumerate(days[wall_at:wall_at + 60]):
    ok, n, dt, err = one(ds)
    if not ok:
        fail += 1; log(f"  fail#{fail} at {i} {ds} {err}")
        if fail >= 3: break
    time.sleep(1.0)
log(f"  D 결과: {i+1}콜 중 실패 {fail}, 경과 {time.time()-t0:.0f}s")
log("PROBE DONE")
