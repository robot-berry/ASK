from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


def canonical(value: str) -> str:
    # NFC keeps meaningful super/subscripts (x², x₁) intact while still
    # canonicalizing composed characters. Full-width punctuation is mapped
    # explicitly below.
    text = unicodedata.normalize("NFC", value)
    text = text.translate(
        str.maketrans(
            {
                "，": ",",
                "；": ";",
                "：": ":",
                "。": ".",
                "−": "-",
                "—": "-",
                "（": "(",
                "）": ")",
                "、": ",",
                "Ａ": "A",
                "Ｂ": "B",
                "Ｃ": "C",
                "Ｄ": "D",
            }
        )
    )
    return re.sub(r"\s+", "", text)


def evaluate(report_path: Path, gold_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    records = {
        Path(str(record["source"])).stem: record
        for record in report.get("records", [])
    }
    details: list[dict[str, object]] = []
    for case in gold["cases"]:
        case_id = str(case["id"])
        transcript = str(records.get(case_id, {}).get("ocr_text") or "")
        compact = canonical(transcript)
        missing = [
            pattern
            for pattern in case.get("required", [])
            if re.search(pattern, compact, re.IGNORECASE) is None
        ]
        forbidden = [
            pattern
            for pattern in case.get("forbidden", [])
            if re.search(pattern, compact, re.IGNORECASE) is not None
        ]
        details.append(
            {
                "id": case_id,
                "pass": bool(transcript) and not missing and not forbidden,
                "missing": missing,
                "forbidden_hits": forbidden,
                "transcript": transcript,
            }
        )
    passed = sum(bool(detail["pass"]) for detail in details)
    total = len(details)
    return {
        "gold_version": gold["version"],
        "report": str(report_path.resolve()),
        "passed": passed,
        "total": total,
        "accuracy_percent": round(100.0 * passed / max(1, total), 2),
        "details": details,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path(__file__).with_name("P_DATA_math_gold_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.report, args.gold)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(
        f"semantic accuracy: {result['passed']}/{result['total']} "
        f"({result['accuracy_percent']:.2f}%)"
    )
    for detail in result["details"]:
        state = "PASS" if detail["pass"] else "FAIL"
        print(f"{detail['id']}: {state}")
        if detail["missing"]:
            print("  missing: " + " | ".join(detail["missing"]))
        if detail["forbidden_hits"]:
            print("  forbidden: " + " | ".join(detail["forbidden_hits"]))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
