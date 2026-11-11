from __future__ import annotations

import json
import sys

from test7_ocr.external_diagnosis import assemble_diagnosis


if __name__ == "__main__":
    print(json.dumps(assemble_diagnosis(), ensure_ascii=False, indent=2))
    raise SystemExit(0)
