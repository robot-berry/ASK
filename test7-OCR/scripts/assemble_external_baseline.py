from __future__ import annotations

import json
import sys

from test7_ocr.external_eval import assemble_external_baseline


if __name__ == "__main__":
    print(json.dumps(assemble_external_baseline(), ensure_ascii=False, indent=2))
    raise SystemExit(0)
