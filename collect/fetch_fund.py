"""KRX 종목별 PER·PBR·EPS·BPS·DIV·DPS 월말 스냅샷 (KRX가 해당 일자에 공표한 값 = 시점 정보).
결과: data/fund_monthly.parquet (date, ticker, BPS, PER, PBR, EPS, DIV, DPS)"""
import time
from pathlib import Path
import pandas as pd
from pykrx import stock
OUT = Path("data"); OUT.mkdir(exist_ok=True)
k = pd.read_parquet(OUT / "kospi.parquet").index
k = pd.to_datetime(k); me = pd.Series(k).groupby(k.to_period("M")).max()
fn = getattr(stock, "get_market_fundamental_by_ticker", None) or stock.get_market_fundamental
parts = []
for d in me.values:
    ds = pd.Timestamp(d).strftime("%Y%m%d")
    for m in ("KOSPI", "KOSDAQ"):
        for a in range(3):
            df = fn(ds, market=m)
            if not df.empty: break
            time.sleep(20)
        if df.empty:
            print("EMPTY", ds, m, flush=True); continue
        df = df.reset_index().rename(columns={df.index.name or "index": "ticker"}); df["date"] = ds; df["market"] = m
        parts.append(df); time.sleep(0.7)
    print(ds, len(parts), flush=True)
F = pd.concat(parts, ignore_index=True)
F.to_parquet(OUT / "fund_monthly.parquet"); print(F.shape, F.columns.tolist())
