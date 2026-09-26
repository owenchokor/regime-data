"""텔레그램 리포트 수신함.
collect : 봇이 받은 파일을 3시간마다 색인(phase3/inbox/index.csv). 원본은 저장·커밋하지 않음(공개 리포, 저작권).
fetch D : 토론 직전 러너에 해당 월 파일만 내려받아 phase3/inbox/D/ 에 둠(커밋 금지, .gitignore).
허용 발신: STASH SIGNALS 방(CHAT_ID) 또는 그 방 멤버의 1:1 대화. 왜: 공개 리포라 봇을 찾은 외부인이 LLM에 문서를 주입하는 경로 차단.
"""
from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

TOKEN, CHAT = os.environ["TG_TOKEN"], os.environ["TG_CHAT"]
API = f"https://api.telegram.org/bot{TOKEN}"
ROOT = Path(__file__).resolve().parent / "inbox"; ROOT.mkdir(exist_ok=True)
IDX = ROOT / "index.csv"; STATE = Path("inbox_state.json")          # state는 release 자산(커밋 소음 방지)
KST = timezone(timedelta(hours=9))
MAX_MB = 20                                                           # 봇 API getFile 한도
COLS = ["update_id", "received_kst", "chat_type", "file_unique_id", "file_id", "file_name", "mime", "size_mb", "caption"]


def tg(method: str, **params) -> dict:
    j = requests.post(f"{API}/{method}", data=params, timeout=60).json()
    if not j.get("ok"):
        print(f"::error::telegram {method}: {j.get('error_code')} {j.get('description')}")
    return j


def reply(chat_id: int, msg_id: int, text: str) -> None:
    tg("sendMessage", chat_id=chat_id, text=text, reply_to_message_id=msg_id)


def read_idx() -> list[dict]:
    import csv
    return list(csv.DictReader(IDX.open(encoding="utf-8"))) if IDX.exists() else []


def write_idx(rows: list[dict]) -> None:
    import csv
    with IDX.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)


def allowed(m: dict, members: dict) -> bool:
    c = m["chat"]
    if str(c["id"]) == str(CHAT):
        return True
    if c["type"] != "private":
        return False
    uid = m["from"]["id"]
    if uid not in members:
        st = tg("getChatMember", chat_id=CHAT, user_id=uid).get("result", {}).get("status", "")
        members[uid] = st in ("creator", "administrator", "member")
    return members[uid]


def collect() -> None:
    st = json.loads(STATE.read_text()) if STATE.exists() else {"offset": 0}
    rows = read_idx(); seen = {r["file_unique_id"] for r in rows}
    j = tg("getUpdates", offset=st["offset"], timeout=0, allowed_updates=json.dumps(["message"]))
    if not j.get("ok"):
        sys.exit(1)                     # 409 = 다른 곳(Colab 파일럿·webhook)이 같은 토큰으로 수신 중
    members, new = {}, 0
    for u in j["result"]:
        st["offset"] = u["update_id"] + 1
        m = u.get("message") or {}
        doc = m.get("document")
        if not doc:
            continue
        if not allowed(m, members):
            print("거부: 비멤버 발신"); continue
        mb = (doc.get("file_size") or 0) / 1e6
        if mb > MAX_MB:
            reply(m["chat"]["id"], m["message_id"], f"⚠️ {MAX_MB}MB 초과({mb:.1f}MB) — 봇이 받을 수 없음. 분할해서 다시 보내주세요."); continue
        if doc["file_unique_id"] in seen:
            continue
        t = datetime.fromtimestamp(m["date"], KST)
        rows.append(dict(update_id=u["update_id"], received_kst=t.strftime("%Y-%m-%d %H:%M"), chat_type=m["chat"]["type"],
                         file_unique_id=doc["file_unique_id"], file_id=doc["file_id"], file_name=doc.get("file_name", ""),
                         mime=doc.get("mime_type", ""), size_mb=round(mb, 2), caption=(m.get("caption") or "").replace("\n", " ")))
        seen.add(doc["file_unique_id"]); new += 1
        reply(m["chat"]["id"], m["message_id"], f"📥 접수: {doc.get('file_name', '')} ({t:%m/%d %H:%M}) — 다음 월말 토론 입력에 포함")
    STATE.write_text(json.dumps(st))
    if new:
        write_idx(rows)
    print(f"updates {len(j['result'])}, 신규 파일 {new}, offset {st['offset']}")


def fetch(D: str) -> None:
    """D월 토론 입력 = 직전 결정(as_of) 다음날 ~ 현재까지 받은 파일. 직전 결정이 없으면 45일."""
    import pandas as pd
    prev = Path("forward/decisions") / f"{pd.Period(D, 'M') - 1}.csv"
    start = (pd.read_csv(prev).as_of.iloc[0] if prev.exists()
             else (datetime.now(KST) - timedelta(days=45)).strftime("%Y-%m-%d"))
    out = ROOT / D; out.mkdir(exist_ok=True)
    rows = [r for r in read_idx() if r["received_kst"][:10] > start]
    L = [f"# {D} 사용자 제공 리포트 ({start} 이후 수신 {len(rows)}건)", "", "| 파일 | 수신(KST) | 캡션 |", "|---|---|---|"]
    for r in rows:
        fp = tg("getFile", file_id=r["file_id"]).get("result", {}).get("file_path")
        if not fp:
            L.append(f"| {r['file_name']} | {r['received_kst']} | 다운로드 실패 |"); continue
        (out / r["file_name"]).write_bytes(requests.get(f"https://api.telegram.org/file/bot{TOKEN}/{fp}", timeout=120).content)
        L.append(f"| {r['file_name']} | {r['received_kst']} | {r['caption']} |")
    (out / "INDEX.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    collect() if sys.argv[1] == "collect" else fetch(sys.argv[2])
