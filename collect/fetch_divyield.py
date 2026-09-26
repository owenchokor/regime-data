"""KOSPI/KOSDAQ 지수 PER·PBR·배당수익률 일별 → 총수익 근사(가격수익 + 배당수익률/252)용.
결과: data/index_fund_{1001,2001}.parquet"""
from pathlib import Path
import pandas as pd
from pykrx import stock
OUT = Path("data"); OUT.mkdir(exist_ok=True)
end = pd.Timestamp.today().strftime("%Y%m%d")
for code in ("1001", "2001"):
    df = stock.get_index_fundamental_by_date("20140101", end, code)
    if df.empty:
        raise RuntimeError(f"index fundamental empty: {code}")
    print(code, df.shape, list(df.columns), df.index[0], df.index[-1])
    df.to_parquet(OUT / f"index_fund_{code}.parquet")
