"""
KRX 지수 일봉 수집 (KOSPI 1001, KOSDAQ 2001)
결과: data/kospi.parquet, data/kosdaq.parquet
"""
import os, sys
from pathlib import Path
import pandas as pd
from pykrx import stock

KRX_ID = os.environ["KRX_ID"]
KRX_PW = os.environ["KRX_PW"]
START = "20140101"
OUT = Path("data"); OUT.mkdir(exist_ok=True)

def fetch(code: str, name: str) -> None:
    end = pd.Timestamp.today().strftime("%Y%m%d")
    df = stock.get_index_ohlcv_by_date(START, end, code)
    if df.empty:
        raise RuntimeError(f"index empty: {code} — 인증 실패 또는 잘못된 코드")
    print(f"{name}: {len(df)}행 {df.index[0]}~{df.index[-1]}")
    df.to_parquet(OUT / f"{name}.parquet")

fetch("1001", "kospi")   # 확인 필요: KRX 지수 코드
fetch("2001", "kosdaq")
