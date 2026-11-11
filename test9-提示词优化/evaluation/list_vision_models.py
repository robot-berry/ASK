"""List vision-capable model ids offered by the configured OpenAI-compatible endpoint."""
import json, os, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
key = (ROOT / "api_key.txt").read_text(encoding="utf-8").strip()
base = os.environ.get("DASHSCOPE_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")

req = urllib.request.Request(base + "/models", headers={"Authorization": "Bearer " + key})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.loads(r.read().decode("utf-8"))

ids = sorted(m["id"] for m in data.get("data", []))
print("total models", len(ids))
hits = [i for i in ids if any(t in i.lower() for t in ("vl", "vision", "glm-4.5v", "-v-", "omni"))]
print("\n".join(hits) if hits else "no vision-looking ids")
