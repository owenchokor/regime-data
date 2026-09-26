"""
추가 데이터 수집 (탐색 겸 수집 — 함수 동작이 미검증이라 결과·오류를 모두 extra_log.json에 기록)
1) 업종 분류: pykrx get_market_sector_classifications — 매월 첫 거래일 스냅샷 (과거 날짜 동작 여부 확인 필요)
2) 지수 목록: 이름에 TR/총수익/변동성/VKOSPI 포함 지수를 찾아 일봉 수집
3) 관리종목 현황: FinanceDataReader KRX-ADMINISTRATIVE (현재 시점만 제공)
결과: data/sector_monthly.parquet, data/index_extra_{code}.parquet, data/admin_snapshot_{YYYYMMDD}.parquet, data/extra_log.json
"""
import json, time, traceback
from pathlib import Path
import pandas as pd
from pykrx import stock

OUT = Path("data"); OUT.mkdir(exist_ok=True)
LOG: dict = {}
def rec(k, v): LOG[k] = v; print(k, str(v)[:300], flush=True)

days = pd.read_parquet(OUT / "kospi.parquet").index
firsts = pd.Series(days, index=days).groupby(days.to_period("M")).first()

# 1) 업종 — 먼저 3개 날짜로 과거 동작 확인
probe = {}
for ds in [firsts.iloc[0], firsts.iloc[len(firsts)//2], firsts.iloc[-1]]:
    d = ds.strftime("%Y%m%d")
    try:
        df = stock.get_market_sector_classifications(d, "KOSPI")
        probe[d] = {"n": len(df), "cols": list(map(str, df.columns)), "head": df.head(3).astype(str).to_dict()}
    except Exception:
        probe[d] = {"error": traceback.format_exc()[-300:]}
    time.sleep(1)
rec("sector_probe", probe)
hist_ok = all(v.get("n", 0) > 0 for v in probe.values())
dates = firsts if hist_ok else firsts.iloc[-1:]
parts = []
for ds in dates:
    d = ds.strftime("%Y%m%d")
    for m in ("KOSPI", "KOSDAQ"):
        try:
            df = stock.get_market_sector_classifications(d, m)
            if not df.empty:
                df = df.copy(); df["market"] = m; df["date"] = d; parts.append(df)
        except Exception as e:
            LOG.setdefault("sector_err", []).append(f"{d} {m} {type(e).__name__}")
        time.sleep(0.7)
if parts:
    sec = pd.concat(parts); sec.index.name = "ticker"
    sec.reset_index().astype({c: str for c in sec.reset_index().columns if sec.reset_index()[c].dtype == object}).to_parquet(OUT / "sector_monthly.parquet")
    rec("sector_saved", {"rows": len(sec), "dates": int(sec.date.nunique()), "hist_ok": hist_ok})

# 2) 지수 목록 탐색
found = {}
for mkt in ["KOSPI", "KOSDAQ", "KRX", "테마"]:
    try:
        for t in stock.get_index_ticker_list(market=mkt):
            nm = stock.get_index_ticker_name(t)
            found[t] = (mkt, nm)
    except Exception:
        LOG[f"idxlist_err_{mkt}"] = traceback.format_exc()[-300:]
    time.sleep(1)
rec("index_all", found)
keys = ["TR", "총수익", "변동성", "VKOSPI", "V-KOSPI"]
cand = {t: v for t, v in found.items() if any(k in v[1] for k in keys)}
rec("index_candidates", cand)
for t, (mkt, nm) in list(cand.items())[:15]:
    try:
        df = stock.get_index_ohlcv_by_date("20140101", pd.Timestamp.today().strftime("%Y%m%d"), t, name_display=False)
        if not df.empty:
            df.to_parquet(OUT / f"index_extra_{t}.parquet")
        LOG.setdefault("index_fetched", {})[t] = [nm, len(df)]
    except Exception as e:
        LOG.setdefault("index_fetched", {})[t] = [nm, f"ERR {type(e).__name__}"]
    time.sleep(1)

# 3) 관리종목 현재 스냅샷
try:
    import FinanceDataReader as fdr
    adm = fdr.StockListing("KRX-ADMINISTRATIVE")
    adm.astype(str).to_parquet(OUT / f"admin_snapshot_{pd.Timestamp.today():%Y%m%d}.parquet")
    rec("admin", {"n": len(adm), "cols": list(adm.columns)})
except Exception:
    rec("admin_err", traceback.format_exc()[-300:])

(OUT / "extra_log.json").write_text(json.dumps(LOG, ensure_ascii=False, indent=1, default=str))
