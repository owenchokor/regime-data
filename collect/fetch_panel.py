"""
전 종목 일별 횡단면 패널 (날짜×시장 1콜) — 체크포인트/재개 지원
소스: stock.get_market_cap_by_ticker(date, market)
결과: data/panel_{close,mcap,volume,shares}.parquet (날짜×티커, 미수정), data/panel_log.txt

재개: release의 기존 panel_close.parquet가 있으면 그 마지막 날짜 이후부터 수집.
체크포인트: CKPT_MIN 분마다 부분 결과를 release에 업로드 → 타임아웃으로 죽어도 진행분 보존.
"""
import os, subprocess, time, traceback
from pathlib import Path
import pandas as pd
from pykrx import stock

OUT = Path("data"); OUT.mkdir(exist_ok=True)
CKPT_MIN = float(os.environ.get("CKPT_MIN", 20))
COLS = {"종가": "close", "시가총액": "mcap", "거래량": "volume", "상장주식수": "shares"}
LOG = open(OUT / "panel_log.txt", "a")

def log(*a):
    m = f"[{time.strftime('%H:%M:%S')}] " + " ".join(map(str, a))
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

def save(buf: dict) -> None:
    for v, dd in buf.items():
        if dd:
            pd.DataFrame(dd).T.sort_index().astype("float32").to_parquet(OUT / f"panel_{v}.parquet")

def upload() -> None:
    f = [str(p) for p in OUT.glob("panel_*")]
    r = subprocess.run(["gh", "release", "upload", "data-latest", *f, "--clobber"],
                       capture_output=True, text=True)
    log("upload rc=", r.returncode, r.stderr.strip()[-200:])

days = pd.read_parquet(OUT / "kospi.parquet").index
buf = {v: {} for v in COLS.values()}
prev = OUT / "panel_close.parquet"
if prev.exists():  # 왜: 타임아웃 재개 — 이미 받은 날짜는 건너뜀
    for v in COLS.values():
        p = OUT / f"panel_{v}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            buf[v] = {d: df.loc[d].dropna() for d in df.index}
    done = set(pd.read_parquet(prev).index)
    days = days[~days.isin(done)]
    log("resume: 기존", len(done), "일 / 남은", len(days), "일")

def call(ds: str, mkt: str) -> pd.DataFrame:
    for _ in range(3):
        try:
            df = stock.get_market_cap_by_ticker(ds, market=mkt)
            if not df.empty: return df
        except Exception:
            log("EXC", ds, mkt, traceback.format_exc()[-300:])
        time.sleep(3)
    return pd.DataFrame()

t0 = last = time.time(); empty = []
for i, dt in enumerate(days):
    ds = dt.strftime("%Y%m%d")
    parts = [p for p in (call(ds, m) for m in ("KOSPI", "KOSDAQ")) if not p.empty]
    if len(parts) < 2:
        empty.append(ds); log("EMPTY", ds, len(parts))
        if i == 4 and len(empty) == 5:
            raise RuntimeError("초반 5일 연속 빈 응답 — 인증/함수 확인")
        if not parts: continue
    df = pd.concat(parts); df = df[~df.index.duplicated()]
    if i == 0: log("columns:", df.columns.tolist(), "n:", len(df))
    for k, v in COLS.items(): buf[v][dt] = df[k]
    if i % 50 == 0:
        rate = (time.time() - t0) / (i + 1)
        log(f"{i}/{len(days)} {ds} n={len(df)} {rate:.2f}s/일 ETA {rate*(len(days)-i)/60:.0f}분")
    if time.time() - last > CKPT_MIN * 60:
        save(buf); upload(); last = time.time(); log("ckpt", i)

save(buf); upload()
log("DONE empty:", len(empty), empty[:20], "elapsed", round((time.time() - t0) / 60, 1), "분")
