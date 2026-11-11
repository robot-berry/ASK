"""Physics uses the same incremental tier publisher as mathematics."""
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "math" / "source" / "tier_streaming.py"
_SPEC = importlib.util.spec_from_file_location("test6_shared_tier_streaming", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(str(_PATH))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
TierStreamPublisher = _MODULE.TierStreamPublisher
