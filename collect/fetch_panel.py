"""
전 종목 일별 횡단면 패널 수집 (날짜 1콜 = 전 종목)
소스: stock.get_market_cap_by_ticker(date, market="ALL") — 컬럼: 종가, 시가총액, 거래량, 거래대금, 상장주식수
거래일 목록은 data/kospi.parquet 인덱스를 사용 (release에서 받아둠)
결과: data/panel_{close,mcap,volume,shares}.parquet (날짜×티커, 미수정)
"""
import time
from pathlib import Path
import pandas as pd
from pykrx import stock

OUT = Path("data"); OUT.mkdir(exist_ok=True)
days = pd.read_parquet(OUT / "kospi.parquet").index
cols = {"종가": "close", "시가총액": "mcap", "거래량": "volume", "상장주식수": "shares"}
buf = {v: {} for v in cols.values()}
t0 = time.time(); empty = 0
for i, dt in enumerate(days):
    ds = dt.strftime("%Y%m%d")
    for attempt in range(3):
        df = stock.get_market_cap_by_ticker(ds, market="ALL")
        if not df.empty: break
        time.sleep(5)
    if df.empty:          # 왜: 인증 실패가 빈 DF로 조용히 옴 → 누락 날짜 기록
        empty += 1; print("EMPTY", ds); continue
    for k, v in cols.items():
        buf[v][dt] = df[k]
    if i % 100 == 0:
        print(f"{i}/{len(days)} {ds} n={len(df)} {time.time()-t0:.0f}s", flush=True)
if empty > 10:
    raise RuntimeError(f"빈 응답 {empty}일")
for v, d in buf.items():
    pd.DataFrame(d).T.sort_index().astype("float32").to_parquet(OUT / f"panel_{v}.parquet")
    print(v, "saved")
print("empty days:", empty, "elapsed", time.time() - t0)
