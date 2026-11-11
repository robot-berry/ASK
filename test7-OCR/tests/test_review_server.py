from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from test7_ocr.review_server import (
    OUTPUT_ISSUE_KEYS,
    ReviewHandler,
    ReviewStore,
    json_hash,
)


class ReviewRouteTests(unittest.TestCase):
    def test_bare_root_redirects_to_actual_ocr_product(self) -> None:
        handler = object.__new__(ReviewHandler)
        handler.path = "/"
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()

        handler.do_GET()

        handler.send_response.assert_called_once_with(302)
        self.assertIn(
            (("Location", "/ocr/"), {}),
            [(call.args, call.kwargs) for call in handler.send_header.call_args_list],
        )
        handler.end_headers.assert_called_once_with()

    def test_legacy_root_review_query_redirects_to_canonical_review_url(self) -> None:
        handler = object.__new__(ReviewHandler)
        handler.path = "/?ui=33&run=full-300&sample=GKP-PHY-0001"
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()

        handler.do_GET()

        handler.send_response.assert_called_once_with(302)
        self.assertIn(
            (
                (
                    "Location",
                    "/review/?ui=33&run=full-300&sample=GKP-PHY-0001",
                ),
                {},
            ),
            [(call.args, call.kwargs) for call in handler.send_header.call_args_list],
        )
        handler.end_headers.assert_called_once_with()


class ReviewStoreTests(unittest.TestCase):

    def test_output_review_can_resolve_an_auxiliary_dataset_without_merging_gold_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_reviews = root / "primary-reviews"
            primary_dataset = root / "primary-dataset"
            auxiliary_reviews = root / "auxiliary-reviews"
            auxiliary_dataset = root / "auxiliary-dataset"
            runs_root = root / "runs"
            output_reviews = root / "output-reviews"
            for path in (
                primary_reviews,
                primary_dataset,
                auxiliary_reviews,
                auxiliary_dataset,
                runs_root,
                output_reviews,
            ):
                path.mkdir()

            sample_id = "AUX-Q01"
            source_path = auxiliary_dataset / "source.png"
            Image.new("RGB", (20, 10), "white").save(source_path)
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            task = {
                "sample_id": sample_id,
                "source": {
                    "image_path": "source.png",
                    "image_sha256": source_sha256,
                    "subject": "physics",
                    "include_in_benchmark": True,
                },
                "gold_annotation": {"question_number": "1"},
                "review": {"priority": "p1", "status": "pending"},
            }
            (auxiliary_reviews / f"{sample_id}.json").write_text(
                json.dumps(task), encoding="utf-8"
            )

            run_id = "aux-run"
            sample_dir = runs_root / run_id / sample_id
            sample_dir.mkdir(parents=True)
            output_path = sample_dir / "output.png"
            Image.new("RGB", (800, 100), "white").save(output_path)
            output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
            report = {
                "run_id": run_id,
                "sample_id": sample_id,
                "source_sha256": source_sha256,
                "output_sha256": output_sha256,
                "output_width": 800,
                "output_height": 100,
                "status": "REVIEW",
            }
            (sample_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

            store = ReviewStore(
                primary_reviews,
                primary_dataset,
                runs_root,
                output_reviews,
                auxiliary_review_dir=auxiliary_reviews,
                auxiliary_dataset_root=auxiliary_dataset,
            )

            self.assertEqual(store.list(), [])
            self.assertEqual(store.source_path(sample_id), source_path)
            self.assertEqual(
                [item["sample_id"] for item in store.list_output_items(run_id)],
                [sample_id],
            )

    def test_failed_content_review_survives_layout_only_rerender(self) -> None:
        reviewed = {
            "display_text": "1. 题干\nA. 甲",
            "visual_assets": [],
        }
        current = {
            "display_text": "1. 题干 A. 甲",
            "visual_assets": [],
        }

        self.assertTrue(
            ReviewStore._failure_evidence_unchanged(
                reviewed,
                current,
                ["semantic_content_incomplete"],
            )
        )

    def test_pixel_sensitive_failure_requires_identical_render(self) -> None:
        document = {"display_text": "1. 题干", "visual_assets": []}

        self.assertFalse(
            ReviewStore._failure_evidence_unchanged(
                document,
                document,
                ["formula_render_error"],
            )
        )

    def test_layout_note_prevents_semantic_failure_carry_forward(self) -> None:
        reviewed = {
            "display_text": "1. 题干\nA. 甲 B. 乙",
            "canonical_text": "1. 题干\nA. 甲 B. 乙",
            "visual_assets": [],
        }
        current = json.loads(json.dumps(reviewed))

        self.assertFalse(
            ReviewStore._failure_evidence_unchanged(
                reviewed,
                current,
                ["semantic_content_incomplete"],
                ["选项C没有换行，排版不正确"],
            )
        )

    def test_formula_style_change_reopens_text_coded_failure(self) -> None:
        reviewed = {
            "display_text": "1. 已知向量 a",
            "canonical_text": r"1. 已知向量 $a$",
            "visual_assets": [],
        }
        current = {
            "display_text": "1. 已知向量 a",
            "canonical_text": r"1. 已知向量 $\boldsymbol{a}$",
            "visual_assets": [],
        }

        self.assertFalse(
            ReviewStore._failure_evidence_unchanged(
                reviewed,
                current,
                ["blank_unit_or_punctuation_missing"],
            )
        )

    def test_handwriting_failure_is_reopened_when_crop_changes(self) -> None:
        reviewed = {
            "display_text": "1. 如图",
            "visual_assets": [
                {
                    "asset_id": "asset_001",
                    "crop_sha256": "1" * 64,
                    "source_bbox": {"x": 1, "y": 2, "width": 3, "height": 4},
                    "extraction": "direct_source_crop",
                    "redraw_used": False,
                }
            ],
        }
        current = json.loads(json.dumps(reviewed))
        current["visual_assets"][0]["crop_sha256"] = "2" * 64

        self.assertFalse(
            ReviewStore._failure_evidence_unchanged(
                reviewed,
                current,
                ["handwriting_visible"],
            )
        )

    def test_list_runs_prefers_the_newest_equally_complete_finalized_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            dataset_root = root / "dataset"
            runs_root = root / "runs"
            output_review_root = root / "output-reviews"
            for path in (review_dir, dataset_root, runs_root, output_review_root):
                path.mkdir()
            for run_id, modified in (("full-old", 1000), ("full-new", 2000)):
                sample = runs_root / run_id / "sample-1"
                sample.mkdir(parents=True)
                Image.new("RGB", (800, 200), "white").save(sample / "output.png")
                (sample / "report.json").write_text("{}", encoding="utf-8")
                (runs_root / run_id / "index.finalized.jsonl").write_text("", encoding="utf-8")
                os.utime(runs_root / run_id, (modified, modified))
            runs = ReviewStore(
                review_dir,
                dataset_root,
                runs_root,
                output_review_root,
            ).list_runs()
            self.assertEqual([run["run_id"] for run in runs], ["full-new", "full-old"])

    def test_list_runs_does_not_default_to_a_latency_failed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            dataset_root = root / "dataset"
            runs_root = root / "runs"
            output_review_root = root / "output-reviews"
            for path in (review_dir, dataset_root, runs_root, output_review_root):
                path.mkdir()
            for run_id, modified, under_target in (
                ("qualified-old", 1000, 1),
                ("slow-new", 2000, 0),
            ):
                run = runs_root / run_id
                sample = run / "sample-1"
                sample.mkdir(parents=True)
                Image.new("RGB", (800, 200), "white").save(sample / "output.png")
                (sample / "report.json").write_text("{}", encoding="utf-8")
                (run / "index.finalized.jsonl").write_text("", encoding="utf-8")
                (run / "batch_summary.json").write_text(
                    json.dumps(
                        {
                            "attempted": 1,
                            "failures": [],
                            "latency_seconds": {"under_target_count": under_target},
                            "mode": (
                                "render_only" if run_id == "slow-new" else "end_to_end"
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
                os.utime(run, (modified, modified))
            runs = ReviewStore(
                review_dir,
                dataset_root,
                runs_root,
                output_review_root,
            ).list_runs()
            self.assertEqual(runs[0]["run_id"], "qualified-old")
            self.assertTrue(runs[0]["default_eligible"])
            self.assertTrue(runs[0]["online_latency_sla_applicable"])
            self.assertFalse(runs[1]["default_eligible"])
            self.assertEqual(runs[1]["measurement_mode"], "render_only")
            self.assertFalse(runs[1]["online_latency_sla_applicable"])

    def test_list_runs_does_not_default_to_latency_guard_aborted_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            dataset_root = root / "dataset"
            runs_root = root / "runs"
            output_review_root = root / "output-reviews"
            for path in (review_dir, dataset_root, runs_root, output_review_root):
                path.mkdir()
            run = runs_root / "latency-aborted"
            sample = run / "sample-1"
            sample.mkdir(parents=True)
            Image.new("RGB", (800, 200), "white").save(sample / "output.png")
            (sample / "report.json").write_text("{}", encoding="utf-8")
            (run / "index.finalized.jsonl").write_text("", encoding="utf-8")
            (run / "batch_summary.json").write_text(
                json.dumps(
                    {
                        "attempted": 1,
                        "failures": [],
                        "latency_seconds": {"under_target_count": 1},
                        "resource_guard": {"status": "CLEAR"},
                        "latency_guard": {"status": "ABORTED"},
                    }
                ),
                encoding="utf-8",
            )

            listed = ReviewStore(
                review_dir,
                dataset_root,
                runs_root,
                output_review_root,
            ).list_runs()

            self.assertFalse(listed[0]["default_eligible"])
            self.assertEqual(listed[0]["latency_guard_status"], "ABORTED")

    def test_list_runs_does_not_default_to_external_eval_only_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            dataset_root = root / "dataset"
            runs_root = root / "runs"
            output_review_root = root / "output-reviews"
            for path in (review_dir, dataset_root, runs_root, output_review_root):
                path.mkdir()
            run = runs_root / "omnidocbench_exam_zh_v1_external"
            sample = run / "sample-1"
            sample.mkdir(parents=True)
            Image.new("RGB", (800, 200), "white").save(sample / "output.png")
            (sample / "report.json").write_text("{}", encoding="utf-8")
            (run / "index.finalized.jsonl").write_text("", encoding="utf-8")
            (run / "batch_summary.json").write_text(
                json.dumps(
                    {
                        "attempted": 1,
                        "failures": [],
                        "latency_seconds": {"under_target_count": 1, "p95": 2.0, "target": 5.0},
                        "resource_guard": {"status": "CLEAR"},
                        "latency_guard": {"status": "CLEAR"},
                        "dataset_role": "external_eval_only",
                        "default_review_eligible": False,
                    }
                ),
                encoding="utf-8",
            )

            listed = ReviewStore(
                review_dir,
                dataset_root,
                runs_root,
                output_review_root,
            ).list_runs()

            self.assertFalse(listed[0]["default_eligible"])

    def test_list_runs_uses_formal_p95_latency_gate_not_every_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            dataset_root = root / "dataset"
            runs_root = root / "runs"
            output_review_root = root / "output-reviews"
            for path in (review_dir, dataset_root, runs_root, output_review_root):
                path.mkdir()
            run = runs_root / "p95-qualified"
            for number in range(100):
                sample = run / f"sample-{number:03d}"
                sample.mkdir(parents=True)
                Image.new("RGB", (800, 200), "white").save(sample / "output.png")
                (sample / "report.json").write_text("{}", encoding="utf-8")
            (run / "index.finalized.jsonl").write_text("", encoding="utf-8")
            (run / "batch_summary.json").write_text(
                json.dumps(
                    {
                        "attempted": 100,
                        "failures": [],
                        "latency_seconds": {
                            "under_target_count": 98,
                            "p95": 3.4,
                            "target": 5.0,
                        },
                        "resource_guard": {"status": "CLEAR"},
                        "latency_guard": {"status": "CLEAR"},
                    }
                ),
                encoding="utf-8",
            )

            listed = ReviewStore(
                review_dir,
                dataset_root,
                runs_root,
                output_review_root,
            ).list_runs()

            self.assertTrue(listed[0]["default_eligible"])

    def test_list_runs_does_not_default_to_resource_conflict_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            dataset_root = root / "dataset"
            runs_root = root / "runs"
            output_review_root = root / "output-reviews"
            for path in (review_dir, dataset_root, runs_root, output_review_root):
                path.mkdir()
            for run_id, modified, guard_status in (
                ("clean-old", 1000, "CLEAR"),
                ("conflicted-new", 2000, "ALLOWED_CONFLICT"),
            ):
                run = runs_root / run_id
                sample = run / "sample-1"
                sample.mkdir(parents=True)
                Image.new("RGB", (800, 200), "white").save(sample / "output.png")
                (sample / "report.json").write_text("{}", encoding="utf-8")
                (run / "index.finalized.jsonl").write_text("", encoding="utf-8")
                (run / "batch_summary.json").write_text(
                    json.dumps(
                        {
                            "attempted": 1,
                            "failures": [],
                            "latency_seconds": {"under_target_count": 1},
                            "resource_guard": {"status": guard_status},
                        }
                    ),
                    encoding="utf-8",
                )
                os.utime(run, (modified, modified))
            runs = ReviewStore(
                review_dir,
                dataset_root,
                runs_root,
                output_review_root,
            ).list_runs()
            self.assertEqual(runs[0]["run_id"], "clean-old")
            self.assertTrue(runs[0]["default_eligible"])
            self.assertFalse(runs[1]["default_eligible"])

    def _ready_task(self) -> dict:
        return {
            "sample_id": "sample-1",
            "source": {},
            "candidate_prelabels": {},
            "prior_manual_evidence": {},
            "gold_annotation": {
                "status": "draft",
                "question_type": "solution",
                "question_number": "1",
                "stem_text": "1. 题干",
                "options": [],
                "canonical_transcript": "1. 题干",
                "formula_status": "human_transcribed",
                "text_blocks": [{"bbox": [0, 0, 10, 10]}],
                "visual_assets": [],
            },
            "review": {
                "status": "pending",
                "annotator_id": None,
                "reviewer_1_id": None,
                "reviewer_2_id": None,
                "adjudicator_id": None,
                "checks": {
                    "text_exact": True,
                    "formula_exact": True,
                    "options_complete": True,
                    "visual_assets_complete": True,
                    "handwriting_excluded": True,
                    "reading_order_correct": True,
                },
                "notes": [],
                "history": [],
                "signoffs": [],
            },
        }

    def _verified_task(self) -> dict:
        return {
            "gold_annotation": {
                "status": "verified",
                "question_type": "solution",
                "stem_text": "1. 题干",
                "options": [],
                "canonical_transcript": "1. 题干",
                "text_blocks": [{"bbox": [0, 0, 10, 10]}],
                "formula_status": "human_transcribed",
                "visual_assets": [],
            },
            "review": {
                "status": "verified",
                "annotator_id": "annotator-a",
                "reviewer_1_id": "reviewer-b",
                "reviewer_2_id": "reviewer-c",
                "adjudicator_id": None,
                "checks": {
                    "text_exact": True,
                    "formula_exact": True,
                    "options_complete": True,
                    "visual_assets_complete": True,
                    "handwriting_excluded": True,
                    "reading_order_correct": True,
                },
            },
        }

    def test_disputed_task_requires_distinct_fourth_adjudicator(self) -> None:
        task = self._verified_task()
        existing = {"review": {"status": "adjudication_required"}}
        with self.assertRaisesRegex(ValueError, "fourth adjudicator"):
            ReviewStore._validate(task, existing)
        task["review"]["adjudicator_id"] = "adjudicator-d"
        ReviewStore._validate(task, existing)

    def test_ordinary_double_review_does_not_require_adjudicator(self) -> None:
        ReviewStore._validate(self._verified_task(), {"review": {"status": "reviewing"}})

    def test_verified_choice_requires_structured_options(self) -> None:
        task = self._verified_task()
        task["gold_annotation"]["question_type"] = "choice"
        with self.assertRaisesRegex(ValueError, "at least two options"):
            ReviewStore._validate(task, {"review": {"status": "reviewing"}})
        task["gold_annotation"]["options"] = [
            {"label": "A", "text": "1"},
            {"label": "B", "text": "2"},
        ]
        ReviewStore._validate(task, {"review": {"status": "reviewing"}})

    def test_role_fields_are_server_owned_during_normal_save(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            review_dir.mkdir()
            task = self._ready_task()
            (review_dir / "sample-1.json").write_text(json.dumps(task), encoding="utf-8")
            updated = json.loads(json.dumps(task))
            updated["review"]["status"] = "annotated"
            updated["review"]["annotator_id"] = "annotator-a"
            updated["review"]["history"] = [{"to_status": "verified", "actor_id": "forged"}]
            with self.assertRaisesRegex(ValueError, "server-owned"):
                ReviewStore(review_dir, root / "dataset").save("sample-1", updated, None)

    def test_server_controlled_three_person_signoff_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            review_dir.mkdir()
            task = self._ready_task()
            (review_dir / "sample-1.json").write_text(json.dumps(task), encoding="utf-8")
            store = ReviewStore(review_dir, root / "dataset")

            task = store.signoff("sample-1", "submit_annotation", "annotator-a", json_hash(task))
            self.assertEqual(task["review"]["status"], "annotated")
            with self.assertRaisesRegex(ValueError, "distinct"):
                store.signoff("sample-1", "approve_reviewer_1", "annotator-a", json_hash(task))
            task = store.signoff("sample-1", "approve_reviewer_1", "reviewer-b", json_hash(task))
            task = store.signoff("sample-1", "approve_reviewer_2", "reviewer-c", json_hash(task))

            self.assertEqual(task["review"]["status"], "verified")
            self.assertEqual(task["gold_annotation"]["status"], "verified")
            self.assertEqual(len(task["review"]["signoffs"]), 3)
            self.assertEqual(
                {task["review"][key] for key in ("annotator_id", "reviewer_1_id", "reviewer_2_id")},
                {"annotator-a", "reviewer-b", "reviewer-c"},
            )

    def test_signed_content_edit_invalidates_active_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            review_dir.mkdir()
            task = self._ready_task()
            (review_dir / "sample-1.json").write_text(json.dumps(task), encoding="utf-8")
            store = ReviewStore(review_dir, root / "dataset")
            task = store.signoff("sample-1", "submit_annotation", "annotator-a", json_hash(task))
            task = store.signoff("sample-1", "approve_reviewer_1", "reviewer-b", json_hash(task))

            updated = json.loads(json.dumps(task))
            updated["gold_annotation"]["stem_text"] = "1. 修正后的题干"
            updated["gold_annotation"]["canonical_transcript"] = "1. 修正后的题干"
            saved = store.save("sample-1", updated, json_hash(task))

            self.assertEqual(saved["review"]["status"], "pending")
            self.assertEqual(saved["gold_annotation"]["status"], "draft")
            self.assertIsNone(saved["review"]["annotator_id"])
            self.assertIsNone(saved["review"]["reviewer_1_id"])
            self.assertEqual(len(saved["review"]["signoffs"]), 2)
            self.assertEqual(saved["review"]["history"][-1]["action"], "content_changed_signoffs_invalidated")

    def test_dispute_requires_distinct_fourth_signoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            review_dir.mkdir()
            task = self._ready_task()
            task["review"]["notes"] = ["复核员对公式转写有分歧"]
            (review_dir / "sample-1.json").write_text(json.dumps(task), encoding="utf-8")
            store = ReviewStore(review_dir, root / "dataset")
            task = store.signoff("sample-1", "submit_annotation", "annotator-a", json_hash(task))
            task = store.signoff("sample-1", "approve_reviewer_1", "reviewer-b", json_hash(task))
            task = store.signoff("sample-1", "request_adjudication", "reviewer-c", json_hash(task))
            with self.assertRaisesRegex(ValueError, "distinct fourth"):
                store.signoff("sample-1", "approve_adjudication", "reviewer-b", json_hash(task))
            task = store.signoff("sample-1", "approve_adjudication", "adjudicator-d", json_hash(task))
            self.assertEqual(task["review"]["status"], "verified")
            self.assertEqual(task["review"]["adjudicator_id"], "adjudicator-d")

    def _output_review(self, verdict: str = "pending") -> dict:
        return {
            "schema_version": "output_quality_review_v1",
            "run_id": "run-1",
            "sample_id": "sample-1",
            "source_sha256": "0" * 64,
            "output_sha256": "1" * 64,
            "verdict": verdict,
            "reviewer_id": "reviewer-a",
            "issues": {key: False for key in OUTPUT_ISSUE_KEYS},
            "notes": [],
            "history": [],
            "updated_at_utc": None,
        }

    def test_output_pass_requires_every_issue_explicitly_absent(self) -> None:
        review = self._output_review("pass")
        ReviewStore._validate_output_review(review)
        review["issues"]["latex_source_visible"] = None
        with self.assertRaisesRegex(ValueError, "explicitly marked absent"):
            ReviewStore._validate_output_review(review)
        review["issues"]["latex_source_visible"] = True
        with self.assertRaisesRegex(ValueError, "explicitly marked absent"):
            ReviewStore._validate_output_review(review)

    def test_output_fail_requires_a_recorded_problem(self) -> None:
        review = self._output_review("fail")
        with self.assertRaisesRegex(ValueError, "At least one output issue"):
            ReviewStore._validate_output_review(review)
        review["issues"]["latex_source_visible"] = True
        ReviewStore._validate_output_review(review)

    def test_output_review_is_hash_bound_and_stored_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "gold-reviews"
            dataset_root = root / "dataset"
            runs_root = root / "runs"
            output_review_root = root / "output-reviews"
            review_dir.mkdir()
            dataset_root.mkdir()
            source = dataset_root / "source.png"
            source.write_bytes(b"source-image")
            import hashlib

            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            task = {
                "sample_id": "sample-1",
                "source": {
                    "image_path": "source.png",
                    "image_sha256": source_hash,
                    "subject": "mathematics",
                    "include_in_benchmark": True,
                },
                "candidate_prelabels": {},
                "prior_manual_evidence": {},
                "gold_annotation": {"status": "draft", "question_number": "1", "visual_assets": []},
                "review": {"status": "pending", "priority": "p1", "checks": {}, "history": []},
            }
            (review_dir / "sample-1.json").write_text(json.dumps(task), encoding="utf-8")
            sample_run = runs_root / "run-1" / "sample-1"
            sample_run.mkdir(parents=True)
            output = sample_run / "output.png"
            Image.new("RGB", (800, 300), "white").save(output)
            output_hash = hashlib.sha256(output.read_bytes()).hexdigest()
            report = {
                "run_id": "run-1",
                "sample_id": "sample-1",
                "source_sha256": source_hash,
                "output_sha256": output_hash,
                "status": "REVIEW",
                "output_width": 800,
                "output_height": 301,
                "elapsed_seconds": 4.2,
                "measurement": {
                    "mode": "render_only",
                    "online_latency_sla_applicable": False,
                },
                "issues": [],
            }
            (sample_run / "report.json").write_text(json.dumps(report), encoding="utf-8")
            store = ReviewStore(review_dir, dataset_root, runs_root, output_review_root)
            with self.assertRaisesRegex(ValueError, "dimensions do not match"):
                store.load_output_review("run-1", "sample-1")
            report["output_height"] = 300
            (sample_run / "report.json").write_text(json.dumps(report), encoding="utf-8")
            loaded = store.load_output_review("run-1", "sample-1")["review"]
            etag = json_hash(loaded)
            loaded["verdict"] = "fail"
            loaded["reviewer_id"] = "reviewer-a"
            loaded["issues"] = {key: False for key in OUTPUT_ISSUE_KEYS}
            loaded["issues"]["semantic_content_incomplete"] = True
            saved = store.save_output_review("run-1", "sample-1", loaded, etag)
            self.assertEqual(saved["verdict"], "fail")
            self.assertTrue((output_review_root / "run-1" / "sample-1.json").is_file())
            self.assertEqual(json.loads((review_dir / "sample-1.json").read_text(encoding="utf-8")), task)
            progress = store.output_progress_csv("run-1").decode("utf-8-sig")
            self.assertIn("human_verdict", progress)
            self.assertIn("sample-1", progress)
            self.assertIn("fail", progress)

            newer_sample = runs_root / "run-2" / "sample-1"
            newer_sample.mkdir(parents=True)
            newer_output = newer_sample / "output.png"
            Image.new("RGB", (800, 320), "#f3f4f6").save(newer_output)
            newer_hash = hashlib.sha256(newer_output.read_bytes()).hexdigest()
            newer_report = {
                **report,
                "run_id": "run-2",
                "output_sha256": newer_hash,
                "output_height": 320,
            }
            (newer_sample / "report.json").write_text(
                json.dumps(newer_report), encoding="utf-8"
            )
            response = store.load_output_review("run-2", "sample-1")
            self.assertEqual(response["review"]["verdict"], "pending")
            self.assertEqual(response["report"]["measurement"]["mode"], "render_only")
            self.assertFalse(
                response["report"]["measurement"]["online_latency_sla_applicable"]
            )
            self.assertEqual(len(response["previous_reviews"]), 1)
            self.assertEqual(response["previous_reviews"][0]["run_id"], "run-1")
            self.assertEqual(response["previous_reviews"][0]["verdict"], "fail")
            self.assertFalse(response["previous_reviews"][0]["same_output"])
            self.assertIsNone(response["previous_reviews"][0]["output_match"])
            listed = store.list_output_items("run-2")
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["previous_review_count"], 1)
            self.assertEqual(listed[0]["previous_fail_count"], 1)
            self.assertFalse(listed[0]["same_output_previous_fail"])
            self.assertEqual(listed[0]["measurement_mode"], "render_only")
            self.assertFalse(listed[0]["online_latency_sla_applicable"])

            repeated_sample = runs_root / "run-3" / "sample-1"
            repeated_sample.mkdir(parents=True)
            repeated_output = repeated_sample / "output.png"
            repeated_output.write_bytes(output.read_bytes())
            repeated_report = {
                **report,
                "run_id": "run-3",
                "output_sha256": output_hash,
                "output_height": 300,
            }
            (repeated_sample / "report.json").write_text(
                json.dumps(repeated_report), encoding="utf-8"
            )
            repeated_response = store.load_output_review("run-3", "sample-1")
            self.assertTrue(repeated_response["previous_reviews"][0]["same_output"])
            self.assertEqual(
                repeated_response["previous_reviews"][0]["output_match"],
                "file_hash",
            )
            self.assertTrue(repeated_response["previous_reviews"][0]["same_source"])
            self.assertEqual(repeated_response["effective_human_verdict"], "fail")
            self.assertEqual(repeated_response["inherited_review"]["run_id"], "run-1")
            repeated_listed = store.list_output_items("run-3")
            self.assertEqual(repeated_listed[0]["human_verdict"], "fail")
            self.assertTrue(repeated_listed[0]["human_review_inherited"])
            self.assertEqual(repeated_listed[0]["human_review_origin"], "inherited")
            repeated_review = repeated_response["review"]
            repeated_etag = json_hash(repeated_review)
            repeated_review["verdict"] = "fail"
            repeated_review["reviewer_id"] = "reviewer-b"
            repeated_review["issues"] = {key: False for key in OUTPUT_ISSUE_KEYS}
            repeated_review["issues"]["semantic_content_incomplete"] = True
            with self.assertRaisesRegex(ValueError, "不要重复验收"):
                store.save_output_review(
                    "run-3",
                    "sample-1",
                    repeated_review,
                    repeated_etag,
                )

            pixel_same_sample = runs_root / "run-4" / "sample-1"
            pixel_same_sample.mkdir(parents=True)
            pixel_same_output = pixel_same_sample / "output.png"
            Image.new("RGB", (800, 300), "white").save(
                pixel_same_output,
                compress_level=0,
            )
            pixel_same_hash = hashlib.sha256(pixel_same_output.read_bytes()).hexdigest()
            self.assertNotEqual(pixel_same_hash, output_hash)
            pixel_same_report = {
                **report,
                "run_id": "run-4",
                "output_sha256": pixel_same_hash,
                "output_height": 300,
            }
            (pixel_same_sample / "report.json").write_text(
                json.dumps(pixel_same_report), encoding="utf-8"
            )
            pixel_same_response = store.load_output_review("run-4", "sample-1")
            self.assertEqual(pixel_same_response["effective_human_verdict"], "fail")
            self.assertTrue(pixel_same_response["previous_reviews"][0]["same_output"])
            self.assertFalse(pixel_same_response["previous_reviews"][0]["same_output_bytes"])
            self.assertTrue(pixel_same_response["previous_reviews"][0]["same_output_pixels"])
            self.assertEqual(
                pixel_same_response["previous_reviews"][0]["output_match"],
                "decoded_pixels",
            )

    def test_candidate_structure_cleans_option_delimiter_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "reviews"
            review_dir.mkdir()
            task = {
                "sample_id": "sample-1",
                "candidate_prelabels": {
                    "suggested_transcript": (
                        "1. 设 $x>0$，则结果为$\n"
                        "$A. $x$$ $B. $x^2$$ $C. $2x$$ $D. $0$$"
                    )
                },
            }
            (review_dir / "sample-1.json").write_text(json.dumps(task), encoding="utf-8")
            structured = ReviewStore(review_dir, root / "dataset").candidate_structure("sample-1")
            self.assertEqual(structured["question_number"], "1")
            self.assertEqual(structured["question_type"], "choice")
            self.assertEqual(len(structured["options"]), 4)
            self.assertEqual(structured["options"][0], {"label": "A", "text": "$x$"})
            self.assertNotIn("\\", structured["display_transcript"])
            self.assertFalse(structured["stem_text"].endswith("\n$"))
            self.assertIn("D. $0$", structured["canonical_transcript"])


if __name__ == "__main__":
    unittest.main()
