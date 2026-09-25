"""
전 종목 일별 횡단면 패널 (날짜×시장 1콜)
소스: stock.get_market_cap_by_ticker(date, market) — 기존 snapshot과 동일 함수·시장별 호출
결과: data/panel_{close,mcap,volume,shares}.parquet (날짜×티커, 미수정), data/panel_log.txt
"""
import sys, time, traceback
from pathlib import Path
import pandas as pd
from pykrx import stock

OUT = Path("data"); OUT.mkdir(exist_ok=True)
LOG = open(OUT / "panel_log.txt", "w")
def log(*a):
    msg = " ".join(map(str, a)); print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()

days = pd.read_parquet(OUT / "kospi.parquet").index
cols = {"종가": "close", "시가총액": "mcap", "거래량": "volume", "상장주식수": "shares"}
buf = {v: {} for v in cols.values()}
t0 = time.time(); empty = []

def call(ds: str, mkt: str) -> pd.DataFrame:
    for _ in range(3):
        try:
            df = stock.get_market_cap_by_ticker(ds, market=mkt)
            if not df.empty: return df
        except Exception:
            log("EXC", ds, mkt, traceback.format_exc()[-400:])
        time.sleep(3)
    return pd.DataFrame()

for i, dt in enumerate(days):
    ds = dt.strftime("%Y%m%d")
    parts = [call(ds, m) for m in ("KOSPI", "KOSDAQ")]
    parts = [p for p in parts if not p.empty]
    if len(parts) < 2:
        empty.append(ds); log("EMPTY", ds, len(parts))
        if i < 5 and len(empty) == i + 1 and i == 4:
            raise RuntimeError("초반 5일 연속 빈 응답 — 인증/함수 확인")
        if len(parts) == 0: continue
    df = pd.concat(parts); df = df[~df.index.duplicated()]
    if i == 0: log("columns:", df.columns.tolist(), "n:", len(df))
    for k, v in cols.items(): buf[v][dt] = df[k]
    if i % 100 == 0: log(f"{i}/{len(days)} {ds} n={len(df)} {time.time()-t0:.0f}s")

for v, d in buf.items():
    pd.DataFrame(d).T.sort_index().astype("float32").to_parquet(OUT / f"panel_{v}.parquet")
log("saved; empty days:", len(empty), empty[:20], "elapsed", round(time.time() - t0))
