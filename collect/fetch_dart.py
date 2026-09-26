"""OpenDART 다중회사 주요계정(fnlttMultiAcnt) 수집 → 재무위험 필터 백테스트용.
point-in-time: rcept_no 앞 8자리 = 접수일. 정정공시 시 원공시 값이 보존되는지는 확인 필요.
결과: data/dart_multi.parquet (long), data/dart_corp.parquet, data/dart_log.json"""
import io, json, os, time, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd, requests

KEY = os.environ["DART_KEY"]
OUT = Path("data"); OUT.mkdir(exist_ok=True)
B = "https://opendart.fss.or.kr/api"
YEARS = range(2015, pd.Timestamp.today().year + 1)
REPRT = {"11013": "Q1", "11012": "H1", "11014": "Q3", "11011": "FY"}
log = {"status": {}, "calls": 0}

z = zipfile.ZipFile(io.BytesIO(requests.get(f"{B}/corpCode.xml", params={"crtfc_key": KEY}, timeout=120).content))
root = ET.fromstring(z.read(z.namelist()[0]))
corp = pd.DataFrame([{k: (e.findtext(k) or "").strip() for k in ("corp_code", "corp_name", "stock_code", "modify_date")} for e in root.iter("list")])
corp = corp[corp.stock_code != ""]
panel = pd.read_parquet(OUT / "panel_close.parquet").columns.astype(str)
corp = corp[corp.stock_code.isin(panel)]
corp.to_parquet(OUT / "dart_corp.parquet")
print("corp mapped", len(corp), "/ panel", len(panel), flush=True)

codes = corp.corp_code.tolist(); rows = []
for y in YEARS:
    for rc in REPRT:
        for i in range(0, len(codes), 100):
            for a in range(3):
                try:
                    j = requests.get(f"{B}/fnlttMultiAcnt.json", timeout=60, params={
                        "crtfc_key": KEY, "corp_code": ",".join(codes[i:i + 100]), "bsns_year": str(y), "reprt_code": rc}).json()
                    break
                except Exception as e:
                    time.sleep(5); j = {"status": "EXC", "message": str(e)[:80]}
            log["calls"] += 1; st = j.get("status"); log["status"][st] = log["status"].get(st, 0) + 1
            if st == "000":
                rows += j["list"]
            elif st not in ("013",):
                print(y, rc, i, st, j.get("message"), flush=True)
                if st in ("020", "010", "011", "012", "901"):   # 한도초과·키 오류 → 중단
                    raise SystemExit(json.dumps(log))
            time.sleep(0.2)
        print(y, rc, "rows", len(rows), flush=True)
df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("dart empty: " + json.dumps(log))
df.to_parquet(OUT / "dart_multi.parquet")
(OUT / "dart_log.json").write_text(json.dumps(log, ensure_ascii=False))
print(df.shape, df.columns.tolist(), log)
