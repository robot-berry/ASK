from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main() -> None:
    rows = list(csv.DictReader((RESULTS / "results.csv").open(encoding="utf-8-sig", newline="")))
    errors: list[dict[str, str]] = []
    totals = {
        "rows": len(rows),
        "png_decoded": 0,
        "page_pdfs_opened": 0,
        "question_pdfs_opened": 0,
        "markdown_opened": 0,
        "question_markdown_opened": 0,
        "question_png_decoded": 0,
    }

    for row in rows:
        name = row["name"]
        checks = {
            "png": RESULTS / "artifacts" / row["output_png"],
            "pdf": RESULTS / "artifacts" / row["output_pdf"],
            "markdown": RESULTS / "artifacts" / row["output_markdown"],
            "questions_pdf": RESULTS / "artifacts" / row["questions_pdf"],
            "questions_markdown": RESULTS / "artifacts" / row["questions_markdown"],
        }
        try:
            image = cv2.imread(str(checks["png"]), cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                raise ValueError("PNG cannot be decoded")
            totals["png_decoded"] += 1
        except Exception as exc:
            errors.append({"name": name, "kind": "png", "error": str(exc)})

        for key, total_key in (("pdf", "page_pdfs_opened"), ("questions_pdf", "question_pdfs_opened")):
            try:
                reader = PdfReader(str(checks[key]))
                if len(reader.pages) < 1:
                    raise ValueError("PDF has no pages")
                totals[total_key] += 1
            except Exception as exc:
                errors.append({"name": name, "kind": key, "error": str(exc)})

        for key, total_key in (("markdown", "markdown_opened"), ("questions_markdown", "question_markdown_opened")):
            try:
                text = checks[key].read_text(encoding="utf-8")
                if not text.strip() or ".png" not in text:
                    raise ValueError("Markdown is empty or has no PNG reference")
                totals[total_key] += 1
            except Exception as exc:
                errors.append({"name": name, "kind": key, "error": str(exc)})

        question_dir = checks["questions_pdf"].parent
        cards = sorted(question_dir.glob("q[0-9][0-9][0-9].png"))
        expected = int(row.get("question_count") or 0)
        if len(cards) != expected:
            errors.append({"name": name, "kind": "question_count", "error": f"expected {expected}, found {len(cards)}"})
        for card in cards:
            image = cv2.imread(str(card), cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                errors.append({"name": name, "kind": "question_png", "error": f"cannot decode {card.name}"})
            else:
                totals["question_png_decoded"] += 1

    report = {
        **totals,
        "expected_each": len(rows),
        "validation_passed": not errors,
        "error_count": len(errors),
        "errors": errors,
    }
    target = RESULTS / "output_validation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
