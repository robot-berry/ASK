from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "ocr_accuracy_300_20260902" / "full_audit_queue"
AUDIT_CSV = ROOT / "full_manual_audit.csv"
DECISIONS_TSV = ROOT / "full_audit_decisions.tsv"


def main() -> None:
    with AUDIT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    with DECISIONS_TSV.open("r", encoding="utf-8-sig", newline="") as handle:
        decisions = list(csv.DictReader(handle, delimiter="\t"))

    decision_by_id: dict[str, dict[str, str]] = {}
    for decision in decisions:
        question_id = (decision.get("question_id") or "").strip()
        if not question_id:
            raise ValueError("发现没有 question_id 的审核记录")
        if question_id in decision_by_id:
            raise ValueError(f"审核记录重复：{question_id}")
        decision_by_id[question_id] = decision

    csv_ids = {row["question_id"] for row in rows}
    unknown = sorted(set(decision_by_id) - csv_ids)
    if unknown:
        raise ValueError(f"审核记录包含未知题号：{unknown}")

    updated = 0
    for row in rows:
        decision = decision_by_id.get(row["question_id"])
        if not decision:
            continue
        status = (decision.get("manual_semantic_complete") or "").strip()
        if status not in {"0", "1"}:
            raise ValueError(f"审核结论非法：{row['question_id']} -> {status!r}")
        error_types = (decision.get("manual_error_types") or "").strip()
        row["manual_semantic_complete"] = status
        row["manual_error_types"] = error_types
        row["manual_error_count"] = str(len([item for item in error_types.split(";") if item]))
        row["manual_notes"] = (decision.get("manual_notes") or "").strip()
        updated += 1

    missing = [row["question_id"] for row in rows if row["manual_semantic_complete"] not in {"0", "1"}]
    if missing:
        raise ValueError(f"仍有 {len(missing)} 题未审核：{missing[:10]}")

    with AUDIT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    complete = sum(row["manual_semantic_complete"] == "1" for row in rows)
    print(f"applied={updated} reviewed={len(rows)} complete={complete} incomplete={len(rows) - complete}")


if __name__ == "__main__":
    main()
