"""
FRED 매크로 지표 수집
- DEXKOUS : 원/달러 환율 (공표 지연 ~5 영업일)
- DGS10   : 미국 10년물 금리
- DGS2    : 미국 2년물 금리
결과: data/fred_{sid}.parquet
"""
import io
from pathlib import Path
import pandas as pd, requests

SERIES = ["DEXKOUS", "DGS10", "DGS2"]
OUT = Path("data"); OUT.mkdir(exist_ok=True)

for sid in SERIES:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True,
                    na_values=".").iloc[:, 0].dropna().rename(sid)
    s.to_frame().to_parquet(OUT / f"fred_{sid}.parquet")
    print(f"{sid}: {len(s)}행 {s.index[0].date()}~{s.index[-1].date()}")
