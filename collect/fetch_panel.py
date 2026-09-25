"""
전 종목 일별 횡단면 패널 (날짜×시장 1콜) — 적응형 스로틀 + 체크포인트/재개 + 진행 패널
소스: stock.get_market_cap_by_ticker(date, market)

호출 한도(실측): 무지연 114콜(분당 ~103콜)에서 벽, 0.5s 간격(분당 ~58콜) 120콜 무사고.
벽은 약 4분 뒤 자동 해제됨. → 기본 SLEEP=1.0s, 벽에 걸리면 300초 대기+재로그인 후
SLEEP 을 0.5s 씩 영구 상향(적응형). 경계값 자체는 미확정.

결과: data/panel_{close,mcap,volume,shares}.parquet (날짜×티커, 미수정)
패널: PROGRESS.md 를 main에 커밋 (PROG_MIN분)   체크포인트: release 업로드 (CKPT_MIN분)
"""
import os, subprocess, time, traceback
from pathlib import Path
import pandas as pd
from pykrx import stock
from pykrx.website.comm import auth

OUT = Path("data"); OUT.mkdir(exist_ok=True)
CKPT_MIN = float(os.environ.get("CKPT_MIN", 15))
PROG_MIN = float(os.environ.get("PROG_MIN", 5))
SLEEP    = float(os.environ.get("SLEEP", 1.0))
COOLDOWN = float(os.environ.get("COOLDOWN", 300))
RUN_URL  = os.environ.get("RUN_URL", "")
COLS = {"종가": "close", "시가총액": "mcap", "거래량": "volume", "상장주식수": "shares"}
LOG = open(OUT / "panel_log.txt", "a")

def log(*a):
    m = f"[{time.strftime('%H:%M:%S')}] " + " ".join(map(str, a))
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

def sh(*c): return subprocess.run(c, capture_output=True, text=True)

def save(buf):
    for v, dd in buf.items():
        if dd: pd.DataFrame(dd).T.sort_index().astype("float32").to_parquet(OUT / f"panel_{v}.parquet")

def upload():
    r = sh("gh", "release", "upload", "data-latest", *[str(p) for p in OUT.glob("panel_*")], "--clobber")
    log("ckpt upload rc=", r.returncode, r.stderr.strip()[-120:])

def relogin():
    try:
        auth.set_auth_session(None); s = auth.get_auth_session()
        log("  relogin ok=", bool(s) and s.is_valid())
    except Exception:
        log("  relogin EXC", traceback.format_exc()[-150:])

STATE = {"sleep": SLEEP, "walls": 0}

def write_progress(i, n, prev, rate, last_date, empty, state="RUNNING"):
    tot = n + prev; dn = i + prev
    pct = dn / tot * 100 if tot else 0
    bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
    md = f"""# build-panel 진행 상황

`{state}` · 갱신 {time.strftime('%Y-%m-%d %H:%M:%S')} UTC

```
{bar} {pct:5.1f}%
```

| 항목 | 값 |
|---|---|
| 수집 완료 | {dn:,} / {tot:,} 거래일 |
| 이번 실행 | {i:,} 일 (이어받기 {prev:,} 일) |
| 속도 | {rate:.2f} 초/일 |
| 남은 예상 | **{(n - i) * rate / 60:.0f} 분** |
| 마지막 처리 | {last_date} |
| 호출 간격 | {STATE['sleep']:.1f} 초 (벽 {STATE['walls']}회) |
| 빈 응답 | {len(empty)} 일 |
| 잡 타임아웃 | 340분 |

{"[실행 로그](" + RUN_URL + ")" if RUN_URL else ""}

<sub>{PROG_MIN}분마다 자동 갱신 · 부분 데이터는 {CKPT_MIN}분마다 release `data-latest` 저장</sub>
"""
    Path("PROGRESS.md").write_text(md)
    for _ in range(3):
        sh("git", "add", "PROGRESS.md")
        sh("git", "commit", "-m", f"progress: {pct:.1f}% ({dn}/{tot})")
        sh("git", "pull", "--rebase", "-q")
        if sh("git", "push", "-q").returncode == 0: return
        time.sleep(2)
    log("progress push 실패")

days = pd.read_parquet(OUT / "kospi.parquet").index
buf = {v: {} for v in COLS.values()}; prev = 0
if (OUT / "panel_close.parquet").exists():
    for v in COLS.values():
        p = OUT / f"panel_{v}.parquet"
        if p.exists():
            df = pd.read_parquet(p); buf[v] = {d: df.loc[d].dropna() for d in df.index}
    done = set(pd.read_parquet(OUT / "panel_close.parquet").index)
    days = days[~days.isin(done)]; prev = len(done)
    log("resume:", prev, "일 완료 /", len(days), "일 남음")

def call(ds: str, mkt: str) -> pd.DataFrame:
    """벽에 걸리면 쿨다운+재로그인 후 재시도, 간격을 영구 상향"""
    for a in range(5):
        try:
            df = stock.get_market_cap_by_ticker(ds, market=mkt)
            if not df.empty: return df
        except Exception as e:
            if a == 0: log(f"  EXC {ds} {mkt} {type(e).__name__}")
        STATE["walls"] += 1
        STATE["sleep"] = min(STATE["sleep"] + 0.5, 5.0)   # 왜: 한도 경계 미확정 → 적응형 상향
        log(f"  WALL {ds} {mkt} try={a} → {COOLDOWN}s 대기, sleep={STATE['sleep']:.1f}s")
        time.sleep(COOLDOWN); relogin()
    return pd.DataFrame()

n = len(days); t0 = time.time(); last_c = last_p = t0; empty = []
write_progress(0, n, prev, 0.0, "-", empty, "START")
try:
    for i, dt in enumerate(days):
        ds = dt.strftime("%Y%m%d")
        parts = []
        for m in ("KOSPI", "KOSDAQ"):
            parts.append(call(ds, m)); time.sleep(STATE["sleep"])
        parts = [p for p in parts if not p.empty]
        if len(parts) < 2:
            empty.append(ds); log("EMPTY", ds, len(parts))
        if parts:
            df = pd.concat(parts); df = df[~df.index.duplicated()]
            if i == 0: log("columns:", df.columns.tolist(), "n:", len(df))
            for k, v in COLS.items(): buf[v][dt] = df[k]
        now = time.time(); rate = (now - t0) / (i + 1)
        if i % 100 == 0: log(f"{i}/{n} {ds} {rate:.2f}s/일 ETA {rate*(n-i)/60:.0f}분")
        if now - last_p > PROG_MIN * 60:       # 왜: 실패해도 패널은 갱신 (이전 버그)
            write_progress(i + 1, n, prev, rate, ds, empty); last_p = now
        if now - last_c > CKPT_MIN * 60:
            save(buf); upload(); last_c = now
finally:
    save(buf); upload()
    done_n = len(buf["close"]) - prev
    write_progress(max(done_n, 0), n, prev, (time.time() - t0) / max(done_n, 1),
                   str(max(buf["close"]).date()) if buf["close"] else "-", empty, "DONE")
log("DONE empty:", len(empty), empty[:20], "elapsed", round((time.time() - t0) / 60, 1), "분")
