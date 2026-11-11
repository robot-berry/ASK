from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from test7_ocr.collection_plan import build_collection_plan_rows
from test7_ocr.collection_workspace import (
    MISSING_FILENAME,
    PLAN_FILENAME,
    READY_BATCH_FILENAME,
    READINESS_FILENAME,
    REGISTRATION_MAP_FILENAME,
    inspect_collection_workspace,
    prepare_collection_workspace,
    register_guided_collection_image,
)
from test7_ocr.self_built_batch import BATCH_HEADERS
from test7_ocr.self_built_intake import SelfBuiltIntakeStore
from test7_ocr.review_server import ReviewStore


class CollectionWorkspaceTests(unittest.TestCase):
    @staticmethod
    def _image_data_url(color: str) -> str:
        stream = io.BytesIO()
        Image.new("RGB", (640, 480), color).save(stream, format="PNG")
        return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")

    def test_prepare_creates_private_working_copy_and_missing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private-collection"
            report = prepare_collection_workspace(root)
            self.assertTrue(report["working_plan_created_this_run"])
            self.assertEqual(report["planned_rows"], 410)
            self.assertEqual(report["files_missing"], 410)
            self.assertFalse(report["ready_for_import_preflight"])
            self.assertTrue((root / PLAN_FILENAME).is_file())
            self.assertTrue((root / READINESS_FILENAME).is_file())
            self.assertTrue((root / MISSING_FILENAME).is_file())
            self.assertTrue((root / READY_BATCH_FILENAME).is_file())
            self.assertEqual(report["ready_batch_rows"], 0)
            self.assertEqual(report["next_guided_row"]["row_id"], "clean-math-001")
            self.assertEqual(len(report["available_guided_rows"]), 110)
            self.assertTrue(
                {row["role"] for row in report["available_guided_rows"]}
                == {"clean_page", "boundary_page"}
            )
            self.assertTrue((root / "images" / "clean" / "math").is_dir())
            self.assertTrue((root / "images" / "boundary" / "double" / "physics").is_dir())

    def test_prepare_preserves_edits_and_reports_incremental_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private-collection"
            prepare_collection_workspace(root)
            rows = build_collection_plan_rows()
            rows[0]["privacy_checked"] = "true"
            rows[0]["contains_personal_data"] = "false"
            image_path = root / rows[0]["image_path"]
            Image.new("RGB", (640, 480), "white").save(image_path)
            plan_path = root / PLAN_FILENAME
            with plan_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=BATCH_HEADERS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            edited = plan_path.read_bytes()

            report = prepare_collection_workspace(root)
            self.assertFalse(report["working_plan_created_this_run"])
            self.assertEqual(plan_path.read_bytes(), edited)
            self.assertFalse(report["template_unchanged"])
            self.assertEqual(report["files_present"], 1)
            self.assertEqual(report["files_missing"], 409)
            self.assertEqual(report["privacy_checked"], 1)
            self.assertEqual(report["personal_data_clear"], 1)
            self.assertEqual(report["role_progress"]["clean_page"]["present"], 1)
            self.assertEqual(report["ready_batch_rows"], 1)
            with (root / READY_BATCH_FILENAME).open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            self.assertFalse(inspect_collection_workspace(root)["ready_for_import_preflight"])

    def test_guided_upload_registers_plan_row_and_resolves_clean_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            workspace = base / "dataset" / "collection"
            store = SelfBuiltIntakeStore(
                base / "dataset",
                base / "records",
                base / "summary.json",
            )
            prepare_collection_workspace(workspace)
            clean, clean_report, clean_row = register_guided_collection_image(
                workspace,
                store,
                row_id="clean-math-001",
                encoded_image=self._image_data_url("white"),
                privacy_checked=True,
                contains_personal_data=False,
                distribution_allowed=False,
            )
            self.assertEqual(clean["role"], "clean_page")
            self.assertEqual(clean_report["files_present"], 1)
            self.assertEqual(clean_report["ready_batch_rows"], 1)
            self.assertEqual(clean_report["guided_registration_count"], 1)
            self.assertEqual(clean_report["next_guided_row"]["row_id"], "photo-math-001-01")
            self.assertIn(
                "photo-math-001-01",
                {row["row_id"] for row in clean_report["available_guided_rows"]},
            )
            self.assertIn(clean_row["storage_mode"], {"hardlink", "copy", "existing"})
            self.assertTrue((workspace / REGISTRATION_MAP_FILENAME).is_file())

            photo, photo_report, photo_row = register_guided_collection_image(
                workspace,
                store,
                row_id="photo-math-001-01",
                encoded_image=self._image_data_url("gray"),
                privacy_checked=True,
                contains_personal_data=False,
                distribution_allowed=False,
            )
            self.assertEqual(photo["role"], "photo_capture")
            self.assertEqual(photo["metadata"]["clean_asset_id"], clean["asset_id"])
            self.assertEqual(photo["metadata"]["conditions"], ["perspective"])
            self.assertEqual(photo_report["files_present"], 2)
            self.assertEqual(photo_report["ready_batch_rows"], 2)
            self.assertEqual(photo_report["guided_registration_count"], 2)
            self.assertEqual(photo_report["next_guided_row"]["row_id"], "photo-math-001-02")
            self.assertTrue((workspace / photo_row["image_path"]).is_file())
            self.assertEqual(len(store.list()), 2)

            with (workspace / PLAN_FILENAME).open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = {row["row_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(rows["clean-math-001"]["privacy_checked"], "true")
            self.assertEqual(rows["clean-math-001"]["contains_personal_data"], "false")
            self.assertEqual(rows["photo-math-001-01"]["distribution_allowed"], "false")

    def test_capture_is_not_exported_before_its_clean_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private-collection"
            prepare_collection_workspace(root)
            rows = build_collection_plan_rows()
            capture = rows[1]
            capture["privacy_checked"] = "true"
            capture["contains_personal_data"] = "false"
            capture_path = root / capture["image_path"]
            Image.new("RGB", (640, 480), "white").save(capture_path)
            with (root / PLAN_FILENAME).open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=BATCH_HEADERS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

            report = prepare_collection_workspace(root)
            self.assertEqual(report["files_present"], 1)
            self.assertEqual(report["ready_batch_rows"], 0)
            self.assertEqual(len(report["blocked_by_clean_dependency"]), 1)
            self.assertEqual(
                report["blocked_by_clean_dependency"][0]["required_clean_row_id"],
                rows[0]["row_id"],
            )

    def test_review_server_exposes_a_bounded_workspace_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            primary_dataset = root / "primary"
            review_dir.mkdir()
            primary_dataset.mkdir()
            store = ReviewStore(
                review_dir,
                primary_dataset,
                runs_root=root / "runs",
                output_review_root=root / "output-reviews",
                mask_task_dir=root / "mask-tasks",
                mask_dir=root / "masks",
                intake_dataset_root=root / "self-built",
                intake_record_dir=root / "intake-records",
                intake_summary_path=root / "intake-summary.json",
                boundary_task_dir=root / "boundary-tasks",
                boundary_manifest_path=root / "boundary-manifest.jsonl",
                boundary_summary_path=root / "boundary-summary.json",
                capture_pair_task_dir=root / "pair-tasks",
                capture_pair_manifest_path=root / "pair-manifest.jsonl",
                capture_pair_summary_path=root / "pair-summary.json",
            )
            public = store.collection_workspace_summary(refresh=True)
            self.assertEqual(public["planned_rows"], 410)
            self.assertEqual(public["files_missing"], 410)
            self.assertEqual(len(public["missing_rows_preview"]), 10)
            self.assertNotIn("missing_rows", public)
            self.assertNotIn("invalid_rows", public)
            self.assertNotIn("incremental_ready_rows", public)
            self.assertNotIn("blocked_by_clean_dependency", public)
            self.assertEqual(len(public["available_guided_rows"]), 110)

            readiness_path = Path(public["readiness_report"])
            stale = json.loads(readiness_path.read_text(encoding="utf-8"))
            stale.pop("next_guided_row")
            stale.pop("available_guided_rows")
            stale.pop("guided_registration_count")
            readiness_path.write_text(json.dumps(stale), encoding="utf-8")
            migrated = store.collection_workspace_summary()
            self.assertEqual(migrated["next_guided_row"]["row_id"], "clean-math-001")
            self.assertEqual(len(migrated["available_guided_rows"]), 110)


if __name__ == "__main__":
    unittest.main()
