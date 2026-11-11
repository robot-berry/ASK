from __future__ import annotations

from collections import Counter
import csv
import io
import unittest

from test7_ocr.collection_plan import (
    COLLECTION_TARGETS,
    build_collection_plan_rows,
    collection_plan_csv,
    collection_plan_summary,
    validate_collection_plan_rows,
)
from test7_ocr.self_built_batch import BATCH_HEADERS


class CollectionPlanTests(unittest.TestCase):
    def test_plan_meets_every_self_built_collection_quota(self) -> None:
        rows = build_collection_plan_rows()
        self.assertEqual(len(rows), 410)
        self.assertEqual(Counter(row["role"] for row in rows), COLLECTION_TARGETS)
        self.assertEqual(
            Counter(
                row["page_layout"]
                for row in rows
                if row["role"] == "boundary_page"
            ),
            {"single_column": 30, "double_column": 30},
        )
        self.assertEqual(
            {row["device_code"] for row in rows if row["device_code"]},
            {"phone-a", "phone-b", "phone-c"},
        )
        self.assertTrue(all(not row["privacy_checked"] for row in rows))
        self.assertTrue(all(not row["contains_personal_data"] for row in rows))
        self.assertTrue(all(row["distribution_allowed"] == "false" for row in rows))

    def test_csv_is_directly_compatible_with_batch_import_headers(self) -> None:
        encoded = collection_plan_csv()
        reader = csv.DictReader(
            io.StringIO(encoded.decode("utf-8-sig"), newline="")
        )
        rows = list(reader)
        self.assertEqual(reader.fieldnames, BATCH_HEADERS)
        self.assertEqual(len(rows), 410)
        validate_collection_plan_rows(rows)
        first_capture = next(row for row in rows if row["role"] == "photo_capture")
        self.assertTrue(first_capture["clean_row_id"].startswith("clean-"))
        self.assertEqual(first_capture["clean_asset_id"], "")

    def test_summary_makes_clear_that_the_template_is_not_collected_data(self) -> None:
        summary = collection_plan_summary()
        self.assertFalse(summary["privacy_declarations_preapproved"])
        self.assertFalse(summary["ready_for_import"])
        self.assertEqual(summary["role_counts"]["handwriting_capture"], 100)
        self.assertGreaterEqual(len(summary["required_before_import"]), 4)

    def test_missing_capture_row_fails_plan_validation(self) -> None:
        rows = build_collection_plan_rows()
        rows.pop(1)
        with self.assertRaisesRegex(ValueError, "410 rows"):
            validate_collection_plan_rows(rows)


if __name__ == "__main__":
    unittest.main()
