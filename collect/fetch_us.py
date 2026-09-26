"""미국 레짐 게이트 후보 (FRED, 키 불필요 fredgraph.csv)
- NASDAQCOM : 나스닥 종합지수 (가격지수)
- DFF       : 연방기금 실효금리
DGS10·DGS2는 daily-collect(fetch_macro)가 이미 수집.
결과: data/fred_{sid}.parquet"""
import io
from pathlib import Path
import pandas as pd, requests

SERIES = ["NASDAQCOM", "DFF"]
OUT = Path("data"); OUT.mkdir(exist_ok=True)
for sid in SERIES:
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60)
    r.raise_for_status()
    s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True, na_values=".").iloc[:, 0].dropna().rename(sid)
    if s.empty:
        raise RuntimeError(f"empty: {sid}")
    s.to_frame().to_parquet(OUT / f"fred_{sid}.parquet")
    print(f"{sid}: {len(s)}행 {s.index[0].date()}~{s.index[-1].date()}")
