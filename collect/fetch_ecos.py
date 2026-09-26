"""
한국은행 ECOS 시장금리 수집 → data/ecos_{stat}.parquet (long: item_code,item_name,time,value)
시크릿 이름이 확정되지 않아 후보 env 중 비어있지 않은 첫 값을 사용 (값은 로그에 출력하지 않음).
"""
import os, json
from pathlib import Path
import pandas as pd, requests

CANDS = ["ECOS_API_KEY", "ECOS_KEY", "BOK_API_KEY", "BOK_ECOS_KEY", "ECOS_API", "ECOS"]
name = next((c for c in CANDS if os.environ.get(c)), None)
if name is None:
    raise RuntimeError("ECOS 키 없음: 후보 시크릿 이름 전부 비어있음 " + str(CANDS))
KEY = os.environ[name]
print("using secret:", name)
OUT = Path("data"); OUT.mkdir(exist_ok=True)
BASE = "https://ecos.bok.or.kr/api"
STATS = {"721Y001": "M", "722Y001": "M"}   # 확인 필요: 시장금리(월) / 기준금리 — 품목 목록을 함께 저장해 검증
log = {"secret": name}

def get(url):
    r = requests.get(url, timeout=60); r.raise_for_status(); return r.json()

for stat, cyc in STATS.items():
    items = get(f"{BASE}/StatisticItemList/{KEY}/json/kr/1/1000/{stat}")
    if "StatisticItemList" not in items:
        log[stat] = {"error": items}; continue
    rows = items["StatisticItemList"]["row"]
    log[stat] = [{k: r.get(k) for k in ("ITEM_CODE", "ITEM_NAME", "CYCLE", "START_TIME", "END_TIME", "STAT_NAME")} for r in rows]
    recs = []
    for it in {r["ITEM_CODE"]: r for r in rows if r.get("CYCLE") == cyc}.values():
        d = get(f"{BASE}/StatisticSearch/{KEY}/json/kr/1/10000/{stat}/{cyc}/201001/203012/{it['ITEM_CODE']}")
        for x in d.get("StatisticSearch", {}).get("row", []):
            recs.append({"item_code": it["ITEM_CODE"], "item_name": x.get("ITEM_NAME1"),
                         "time": x["TIME"], "value": pd.to_numeric(x["DATA_VALUE"], errors="coerce")})
    df = pd.DataFrame(recs)
    if df.empty:
        raise RuntimeError(f"ECOS empty: {stat}")
    df.to_parquet(OUT / f"ecos_{stat}.parquet")
    print(stat, len(df), "rows,", df.item_code.nunique(), "items")
(OUT / "ecos_items.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
