from __future__ import annotations

from collections import Counter
import csv
import io
from pathlib import Path
from typing import Any

from .self_built_batch import BATCH_HEADERS


COLLECTION_PLAN_VERSION = "self_built_collection_plan_v1"
COLLECTION_TARGETS = {
    "clean_page": 50,
    "photo_capture": 200,
    "handwriting_capture": 100,
    "boundary_page": 60,
}
DEVICE_CODES = ("phone-a", "phone-b", "phone-c")
ORDINARY_CAPTURE_CONDITIONS = (
    "perspective",
    "shadow",
    "fold_or_curve",
    "blur",
)


def _row(**values: object) -> dict[str, str]:
    row = {field: "" for field in BATCH_HEADERS}
    for field, value in values.items():
        if field not in row:
            raise ValueError(f"Unknown collection-plan field: {field}")
        row[field] = str(value)
    return row


def _subject_page(index: int, half: int) -> tuple[str, str, int]:
    if index <= half:
        return "mathematics", "math", index
    return "physics", "physics", index - half


def build_collection_plan_rows() -> list[dict[str, str]]:
    """Build a deterministic, import-compatible plan without claiming data exists."""
    rows: list[dict[str, str]] = []
    capture_index = 0
    for page_number in range(1, 51):
        subject, subject_code, local_index = _subject_page(page_number, 25)
        clean_row_id = f"clean-{subject_code}-{local_index:03d}"
        document_group = f"selfbuilt-{subject_code}-paper-{local_index:03d}"
        rows.append(
            _row(
                row_id=clean_row_id,
                image_path=f"images/clean/{subject_code}/clean-{subject_code}-{local_index:03d}.png",
                role="clean_page",
                subject=subject,
                scope="high_school",
                document_group=document_group,
                page_index=1,
                page_layout="not_applicable",
                provenance="self_created",
                distribution_allowed="false",
            )
        )
        for shot_number, condition in enumerate(ORDINARY_CAPTURE_CONDITIONS, start=1):
            device_code = DEVICE_CODES[capture_index % len(DEVICE_CODES)]
            capture_index += 1
            rows.append(
                _row(
                    row_id=f"photo-{subject_code}-{local_index:03d}-{shot_number:02d}",
                    image_path=(
                        f"images/photo/{subject_code}/photo-{subject_code}-{local_index:03d}-"
                        f"{shot_number:02d}-{condition}-{device_code}.jpg"
                    ),
                    role="photo_capture",
                    subject=subject,
                    scope="high_school",
                    document_group=document_group,
                    page_index=1,
                    clean_row_id=clean_row_id,
                    device_code=device_code,
                    conditions=condition,
                    page_layout="not_applicable",
                    provenance="self_created",
                    distribution_allowed="false",
                )
            )
        handwriting_conditions = ("frontal|uneven_light", "perspective|shadow")
        for shot_number, conditions in enumerate(handwriting_conditions, start=1):
            device_code = DEVICE_CODES[capture_index % len(DEVICE_CODES)]
            capture_index += 1
            rows.append(
                _row(
                    row_id=f"hand-{subject_code}-{local_index:03d}-{shot_number:02d}",
                    image_path=(
                        f"images/handwriting/{subject_code}/hand-{subject_code}-{local_index:03d}-"
                        f"{shot_number:02d}-{device_code}.jpg"
                    ),
                    role="handwriting_capture",
                    subject=subject,
                    scope="high_school",
                    document_group=document_group,
                    page_index=1,
                    clean_row_id=clean_row_id,
                    device_code=device_code,
                    conditions=conditions,
                    page_layout="not_applicable",
                    provenance="self_created",
                    distribution_allowed="false",
                )
            )

    for layout in ("single_column", "double_column"):
        layout_code = "single" if layout == "single_column" else "double"
        for page_number in range(1, 31):
            subject, subject_code, local_index = _subject_page(page_number, 15)
            rows.append(
                _row(
                    row_id=f"boundary-{layout_code}-{subject_code}-{local_index:03d}",
                    image_path=(
                        f"images/boundary/{layout_code}/{subject_code}/"
                        f"boundary-{layout_code}-{subject_code}-{local_index:03d}.jpg"
                    ),
                    role="boundary_page",
                    subject=subject,
                    scope="high_school",
                    document_group=(
                        f"selfbuilt-boundary-{layout_code}-{subject_code}-{local_index:03d}"
                    ),
                    page_index=1,
                    page_layout=layout,
                    provenance="self_created",
                    distribution_allowed="false",
                )
            )
    validate_collection_plan_rows(rows)
    return rows


def validate_collection_plan_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != sum(COLLECTION_TARGETS.values()):
        raise ValueError("Collection plan does not contain exactly 410 rows")
    if any(list(row) != BATCH_HEADERS for row in rows):
        raise ValueError("Collection-plan columns do not match the batch importer")
    row_ids = [row["row_id"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Collection plan contains duplicate row ids")
    role_counts = Counter(row["role"] for row in rows)
    if dict(role_counts) != COLLECTION_TARGETS:
        raise ValueError("Collection plan role quotas are incomplete")
    if any(
        not row["image_path"]
        or Path(row["image_path"]).is_absolute()
        or ".." in Path(row["image_path"]).parts
        for row in rows
    ):
        raise ValueError("Collection plan contains an unsafe image path")
    if any(row["privacy_checked"] or row["contains_personal_data"] for row in rows):
        raise ValueError("Collection plan must not pre-approve privacy declarations")
    if any(row["distribution_allowed"] != "false" for row in rows):
        raise ValueError("Collection plan must default to no public distribution")

    clean_rows = {row["row_id"] for row in rows if row["role"] == "clean_page"}
    clean_refs = Counter(
        row["clean_row_id"]
        for row in rows
        if row["role"] in {"photo_capture", "handwriting_capture"}
    )
    if set(clean_refs) != clean_rows:
        raise ValueError("Every capture must reference one planned clean page")
    for clean_id in clean_rows:
        photos = sum(
            row["role"] == "photo_capture" and row["clean_row_id"] == clean_id
            for row in rows
        )
        handwriting = sum(
            row["role"] == "handwriting_capture" and row["clean_row_id"] == clean_id
            for row in rows
        )
        if photos != 4 or handwriting != 2:
            raise ValueError("Each clean page requires four photos and two handwriting captures")

    captures = [
        row for row in rows if row["role"] in {"photo_capture", "handwriting_capture"}
    ]
    if {row["device_code"] for row in captures} != set(DEVICE_CODES):
        raise ValueError("Collection plan does not cover three anonymous devices")
    covered_conditions = {
        condition
        for row in captures
        for condition in row["conditions"].split("|")
        if condition
    }
    if not set(ORDINARY_CAPTURE_CONDITIONS).issubset(covered_conditions):
        raise ValueError("Collection plan does not cover every mandatory capture condition")
    boundary_layouts = Counter(
        row["page_layout"] for row in rows if row["role"] == "boundary_page"
    )
    if boundary_layouts != {"single_column": 30, "double_column": 30}:
        raise ValueError("Collection plan must contain 30 single- and 30 double-column pages")
    for role, expected_per_subject in {
        "clean_page": 25,
        "photo_capture": 100,
        "handwriting_capture": 50,
        "boundary_page": 30,
    }.items():
        counts = Counter(row["subject"] for row in rows if row["role"] == role)
        if counts != {"mathematics": expected_per_subject, "physics": expected_per_subject}:
            raise ValueError(f"Collection plan {role} is not balanced by subject")


def collection_plan_summary(rows: list[dict[str, str]] | None = None) -> dict[str, Any]:
    planned = rows if rows is not None else build_collection_plan_rows()
    validate_collection_plan_rows(planned)
    captures = [
        row for row in planned if row["role"] in {"photo_capture", "handwriting_capture"}
    ]
    return {
        "schema_version": COLLECTION_PLAN_VERSION,
        "planned_rows": len(planned),
        "role_counts": dict(sorted(Counter(row["role"] for row in planned).items())),
        "subject_counts": dict(sorted(Counter(row["subject"] for row in planned).items())),
        "device_counts": dict(sorted(Counter(row["device_code"] for row in captures).items())),
        "condition_counts": dict(
            sorted(
                Counter(
                    condition
                    for row in captures
                    for condition in row["conditions"].split("|")
                    if condition
                ).items()
            )
        ),
        "boundary_layout_counts": dict(
            sorted(
                Counter(
                    row["page_layout"]
                    for row in planned
                    if row["role"] == "boundary_page"
                ).items()
            )
        ),
        "privacy_declarations_preapproved": False,
        "ready_for_import": False,
        "required_before_import": [
            "place real images at each planned relative image_path",
            "set privacy_checked to true only after inspecting every image",
            "set contains_personal_data explicitly to true or false",
            "change filenames or metadata if the actual capture differs from the plan",
            "run the batch importer in dry-run mode before --apply",
        ],
    }


def collection_plan_csv(rows: list[dict[str, str]] | None = None) -> bytes:
    planned = rows if rows is not None else build_collection_plan_rows()
    validate_collection_plan_rows(planned)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=BATCH_HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(planned)
    return ("\ufeff" + stream.getvalue()).encode("utf-8")
