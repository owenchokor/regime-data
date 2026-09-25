"""
KRX 호출 한도 진단 v2 — 목표: 벽을 피하는 지속 가능 호출 간격 d 찾기
0) 현재 벽 상태  1) 재로그인 복구  2) 대기 복구  3) d=0.5/1.5/3.0s 로 각 120콜 지속성
결과: data/probe_log.txt
"""
import time, traceback
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
cur = 0
def nxt() -> str:
    global cur; cur = (cur + 1) % len(days); return days[cur]

def one(ds: str):
    t = time.time()
    try:
        df = stock.get_market_cap_by_ticker(ds, market="KOSPI")
        return (not df.empty), len(df), time.time() - t, ""
    except Exception as e:
        return False, 0, time.time() - t, f"{type(e).__name__}"

def relogin() -> bool:
    try:
        auth.set_auth_session(None)
        s = auth.get_auth_session()
        ok = bool(s) and s.is_valid()
        log("  relogin ok=", ok); return ok
    except Exception:
        log("  relogin EXC", traceback.format_exc()[-200:]); return False

def sustain(d: float, n: int) -> tuple[int, int]:
    """간격 d 로 n콜 — (성공, 첫실패인덱스)"""
    ok_n, first_fail = 0, -1
    for i in range(n):
        ok, _, _, _ = one(nxt())
        if ok: ok_n += 1
        elif first_fail < 0:
            first_fail = i; log(f"    d={d}: 첫 실패 @{i}")
        time.sleep(d)
    log(f"  d={d}s: {ok_n}/{n} 성공, 첫실패={first_fail}")
    return ok_n, first_fail

try:
    log("=== 0: 현재 벽 상태 ===")
    ok, n, dt, err = one(days[0]); log(f"  ok={ok} n={n} {dt:.2f}s {err}")

    if not ok:
        log("=== 1: 재로그인 복구 ===")
        relogin(); ok, n, _, err = one(days[0]); log(f"  재로그인 후 ok={ok} n={n} {err}")
    if not ok:
        log("=== 2: 대기 복구 ===")
        for w in (30, 60, 120, 300):
            log(f"  sleep {w}s"); time.sleep(w)
            ok, n, _, err = one(days[0]); log(f"  {w}s 후 ok={ok} n={n} {err}")
            if ok: break
    if not ok:
        log("  복구 실패 — 대기/재로그인으로 안 풀림"); raise SystemExit(0)

    log("=== 3: 지속 가능 간격 탐색 ===")
    for d in (0.5, 1.5, 3.0):
        ok_n, ff = sustain(d, 120)
        if ff < 0:
            log(f"  >>> d={d}s 는 120콜 무사고 — 이 간격 채택 가능"); break
        log("  복구 대기 120s"); time.sleep(120); relogin()
except Exception:
    log("FATAL", traceback.format_exc()[-500:])
log("PROBE DONE")
