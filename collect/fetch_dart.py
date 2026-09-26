"""OpenDART 재수집 (이어받기·중간저장·진행기록).
 1) 주요사항보고서 목록(list.json, pblntf_ty=B): 딜 종목(주식교환·합병·양수도 등) 필터용 → data/dart_events.parquet
 2) 다중회사 주요계정(fnlttMultiAcnt): 재무·성장 → data/dart_multi.parquet (최근 연도부터)
point-in-time: rcept_no 앞 8자리 = 접수일. 정정공시 시 원공시 값 보존 여부는 확인 필요.
상태: data/dart_state.json (완료 작업키) — release에서 받아 재시작 시 건너뜀."""
import io, json, os, subprocess, time, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd, requests

KEY = os.environ["DART_KEY"]; OUT = Path("data"); OUT.mkdir(exist_ok=True)
B = "https://opendart.fss.or.kr/api"
BUDGET_MIN = float(os.environ.get("BUDGET_MIN", 95)); T0 = time.time()
ST = json.loads((OUT / "dart_state.json").read_text()) if (OUT / "dart_state.json").exists() else {"done": [], "status": {}, "calls": 0, "sec": 0.0}
done = set(ST["done"])
EV = [pd.read_parquet(OUT / "dart_events.parquet")] if (OUT / "dart_events.parquet").exists() else []
MU = [pd.read_parquet(OUT / "dart_multi.parquet")] if (OUT / "dart_multi.parquet").exists() else []
last_save = last_prog = time.time()

def sh(*c): return subprocess.run(c, capture_output=True, text=True)

def save(final=False):
    if EV: pd.concat(EV, ignore_index=True).drop_duplicates().to_parquet(OUT / "dart_events.parquet")
    if MU: pd.concat(MU, ignore_index=True).drop_duplicates().to_parquet(OUT / "dart_multi.parquet")
    ST["done"] = sorted(done); (OUT / "dart_state.json").write_text(json.dumps(ST))
    sh("gh", "release", "upload", "data-latest", *[str(p) for p in OUT.glob("dart_*")], "--clobber")

def progress(stage, note=""):
    el = (time.time() - T0) / 60
    md = (f"# fetch-dart 진행\n\n갱신 {time.strftime('%Y-%m-%d %H:%M:%S')} UTC · 단계 `{stage}` · 경과 {el:.0f}분 / 예산 {BUDGET_MIN:.0f}분\n\n"
          f"| 항목 | 값 |\n|---|---|\n| 완료 작업 | {len(done)} |\n| 누적 호출 | {ST['calls']} |\n| 평균 응답 | {ST['sec']/max(ST['calls'],1):.2f}초 |\n"
          f"| status 분포 | {ST['status']} |\n| 이벤트 행 | {sum(len(x) for x in EV)} |\n| 재무 행 | {sum(len(x) for x in MU)} |\n\n{note}\n")
    Path("PROGRESS_dart.md").write_text(md)
    for _ in range(3):
        sh("git", "add", "PROGRESS_dart.md"); sh("git", "commit", "-m", f"progress(dart): {stage} {len(done)}")
        sh("git", "pull", "--rebase", "-q")
        if sh("git", "push", "-q").returncode == 0: return

def call(ep, params):
    for a in range(3):
        t = time.time()
        try:
            j = requests.get(f"{B}/{ep}", params={"crtfc_key": KEY, **params}, timeout=40).json()
        except Exception as e:
            j = {"status": "EXC", "message": str(e)[:100]}
        ST["calls"] += 1; ST["sec"] += time.time() - t
        s = j.get("status"); ST["status"][s] = ST["status"].get(s, 0) + 1
        if s in ("000", "013"): return j
        if s in ("020", "010", "011", "012", "901"):
            save(); progress("STOP", f"중단 status={s} {j.get('message')}"); raise SystemExit(f"stop {s}")
        time.sleep(3 * (a + 1))
    return j

def tick(stage):
    global last_save, last_prog
    if time.time() - last_save > 600: save(); last_save = time.time()
    if time.time() - last_prog > 300: progress(stage); last_prog = time.time()
    if (time.time() - T0) / 60 > BUDGET_MIN:
        save(); progress(stage, "예산 소진 — 재실행하면 이어받음"); raise SystemExit(0)

progress("START")
# 1) 주요사항보고서 목록: 3개월 창 × 페이지
end = pd.Timestamp.today()
wins = pd.date_range("2015-01-01", end, freq="3MS")
for w0 in wins[::-1]:
    w1 = min(w0 + pd.DateOffset(months=3) - pd.Timedelta(days=1), end)
    key = f"ev:{w0:%Y%m}"
    if key in done: continue
    page = 1
    while True:
        j = call("list.json", {"bgn_de": f"{w0:%Y%m%d}", "end_de": f"{w1:%Y%m%d}", "pblntf_ty": "B", "page_no": page, "page_count": 100})
        if j.get("status") == "000": EV.append(pd.DataFrame(j["list"]))
        tp = int(j.get("total_page", 0) or 0)
        tick("events")
        if page >= tp: break
        page += 1
    done.add(key); print(key, "pages", page, flush=True)
save(); progress("events done")
# 2) 다중회사 주요계정
z = zipfile.ZipFile(io.BytesIO(requests.get(f"{B}/corpCode.xml", params={"crtfc_key": KEY}, timeout=120).content))
root = ET.fromstring(z.read(z.namelist()[0]))
corp = pd.DataFrame([{k: (e.findtext(k) or "").strip() for k in ("corp_code", "corp_name", "stock_code")} for e in root.iter("list")])
panel = pd.read_parquet(OUT / "panel_close.parquet").columns.astype(str)
corp = corp[(corp.stock_code != "") & corp.stock_code.isin(panel)]; corp.to_parquet(OUT / "dart_corp.parquet")
codes = corp.corp_code.tolist()
for y in range(end.year, 2014, -1):
    for rc in ("11011", "11014", "11012", "11013"):
        for i in range(0, len(codes), 100):
            key = f"fs:{y}:{rc}:{i}"
            if key in done: continue
            j = call("fnlttMultiAcnt.json", {"corp_code": ",".join(codes[i:i + 100]), "bsns_year": str(y), "reprt_code": rc})
            if j.get("status") == "000": MU.append(pd.DataFrame(j["list"]))
            done.add(key); tick(f"fs {y} {rc}")
        print(y, rc, flush=True)
save(); progress("DONE")
