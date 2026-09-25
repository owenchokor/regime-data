"""
전 종목 일별 횡단면 패널 (날짜×시장 1콜) — 체크포인트/재개 + 진행 패널
소스: stock.get_market_cap_by_ticker(date, market)
결과: data/panel_{close,mcap,volume,shares}.parquet (날짜×티커, 미수정)
진행 패널: PROGRESS.md 를 main에 커밋 (PROG_MIN분마다) → 리포에서 실시간 확인
체크포인트: CKPT_MIN분마다 부분 parquet 를 release에 업로드 → 타임아웃해도 보존
"""
import os, subprocess, time, traceback
from pathlib import Path
import pandas as pd
from pykrx import stock

OUT = Path("data"); OUT.mkdir(exist_ok=True)
CKPT_MIN = float(os.environ.get("CKPT_MIN", 15))
PROG_MIN = float(os.environ.get("PROG_MIN", 5))
RUN_URL  = os.environ.get("RUN_URL", "")
COLS = {"종가": "close", "시가총액": "mcap", "거래량": "volume", "상장주식수": "shares"}
LOG = open(OUT / "panel_log.txt", "a")

def log(*a):
    m = f"[{time.strftime('%H:%M:%S')}] " + " ".join(map(str, a))
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

def sh(*c, **k):
    return subprocess.run(c, capture_output=True, text=True, **k)

def save(buf):
    for v, dd in buf.items():
        if dd: pd.DataFrame(dd).T.sort_index().astype("float32").to_parquet(OUT / f"panel_{v}.parquet")

def upload():
    r = sh("gh", "release", "upload", "data-latest", *[str(p) for p in OUT.glob("panel_*")], "--clobber")
    log("ckpt upload rc=", r.returncode, r.stderr.strip()[-150:])

def write_progress(i, n, done_prev, rate, last_date, empty, state="RUNNING"):
    pct = (i + done_prev) / (n + done_prev) * 100
    bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
    eta = (n - i) * rate / 60
    md = f"""# build-panel 진행 상황

`{state}` · 갱신 {time.strftime('%Y-%m-%d %H:%M:%S')} UTC

```
{bar} {pct:5.1f}%
```

| 항목 | 값 |
|---|---|
| 수집 완료 | {i + done_prev:,} / {n + done_prev:,} 거래일 |
| 이번 실행 | {i:,} 일 (이어받기 {done_prev:,} 일) |
| 속도 | {rate:.2f} 초/일 |
| 남은 예상 | **{eta:.0f} 분** |
| 마지막 처리 | {last_date} |
| 빈 응답 | {len(empty)} 일 |
| 잡 타임아웃 | 340분 |

{"[실행 로그](" + RUN_URL + ")" if RUN_URL else ""}

<sub>PROG_MIN={PROG_MIN}분마다 자동 갱신. 부분 데이터는 {CKPT_MIN}분마다 release `data-latest`에 저장됨.</sub>
"""
    Path("PROGRESS.md").write_text(md)
    for _ in range(3):  # 왜: 동시 커밋 충돌 대비 rebase 재시도
        sh("git", "add", "PROGRESS.md")
        sh("git", "commit", "-m", f"progress: {pct:.1f}% ({i + done_prev}/{n + done_prev})")
        sh("git", "pull", "--rebase", "-q")
        if sh("git", "push", "-q").returncode == 0: return
        time.sleep(2)
    log("progress push 실패")

days = pd.read_parquet(OUT / "kospi.parquet").index
buf = {v: {} for v in COLS.values()}; done_prev = 0
if (OUT / "panel_close.parquet").exists():
    for v in COLS.values():
        p = OUT / f"panel_{v}.parquet"
        if p.exists():
            df = pd.read_parquet(p); buf[v] = {d: df.loc[d].dropna() for d in df.index}
    done = set(pd.read_parquet(OUT / "panel_close.parquet").index)
    days = days[~days.isin(done)]; done_prev = len(done)
    log("resume: 기존", done_prev, "일 / 남은", len(days), "일")

def call(ds, mkt):
    for _ in range(3):
        try:
            df = stock.get_market_cap_by_ticker(ds, market=mkt)
            if not df.empty: return df
        except Exception:
            log("EXC", ds, mkt, traceback.format_exc()[-300:])
        time.sleep(3)
    return pd.DataFrame()

n = len(days); t0 = time.time(); last_c = last_p = t0; empty = []
write_progress(0, n, done_prev, 0.0, "-", empty, "START")
try:
    for i, dt in enumerate(days):
        ds = dt.strftime("%Y%m%d")
        parts = [p for p in (call(ds, m) for m in ("KOSPI", "KOSDAQ")) if not p.empty]
        if len(parts) < 2:
            empty.append(ds); log("EMPTY", ds, len(parts))
            if i == 4 and len(empty) == 5: raise RuntimeError("초반 5일 연속 빈 응답 — 인증/함수 확인")
            if not parts: continue
        df = pd.concat(parts); df = df[~df.index.duplicated()]
        if i == 0: log("columns:", df.columns.tolist(), "n:", len(df))
        for k, v in COLS.items(): buf[v][dt] = df[k]
        now = time.time(); rate = (now - t0) / (i + 1)
        if i % 50 == 0: log(f"{i}/{n} {ds} n={len(df)} {rate:.2f}s/일 ETA {rate*(n-i)/60:.0f}분")
        if now - last_p > PROG_MIN * 60:
            write_progress(i + 1, n, done_prev, rate, ds, empty); last_p = now
        if now - last_c > CKPT_MIN * 60:
            save(buf); upload(); last_c = now; log("ckpt", i)
finally:
    save(buf); upload()
    r = (time.time() - t0) / max(len(buf["close"]) - done_prev, 1)
    write_progress(len(buf["close"]) - done_prev, n, done_prev, r,
                   str(max(buf["close"])) if buf["close"] else "-", empty, "DONE")
log("DONE empty:", len(empty), empty[:20], "elapsed", round((time.time()-t0)/60, 1), "분")
