"""claude-code-action execution_file에서 세션별 소요시간·턴·토큰·API환산비용 추출.
total_cost_usd는 API 단가 환산치(구독 과금 아님) — 세션 간 상대 규모 비교용."""
import json, os, sys
from pathlib import Path

out = {}
for k in ("G", "V", "M", "S"):
    f = os.environ.get(f"F_{k}")
    if not f or not Path(f).exists():
        out[k] = None; continue
    try:
        d = json.loads(Path(f).read_text())
        r = next((x for x in reversed(d) if isinstance(x, dict) and x.get("type") == "result"), {})
        u = r.get("usage") or {}
        out[k] = dict(minutes=round((r.get("duration_ms") or 0) / 6e4, 1), turns=r.get("num_turns"),
                      subtype=r.get("subtype"), usd_api_equiv=r.get("total_cost_usd"),
                      input_tokens=u.get("input_tokens"), output_tokens=u.get("output_tokens"),
                      cache_read=u.get("cache_read_input_tokens"), cache_write=u.get("cache_creation_input_tokens"))
    except Exception as e:
        out[k] = {"error": str(e)[:200]}
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
