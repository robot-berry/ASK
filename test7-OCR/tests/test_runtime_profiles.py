from __future__ import annotations

import copy
from pathlib import Path
import unittest

from test7_ocr.cli import build_parser
from test7_ocr.pipeline import load_config
from test7_ocr.runtime_profiles import (
    apply_runtime_profile,
    authenticated_source_sample_id,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RuntimeProfileTests(unittest.TestCase):
    def test_product_v7_profile_matches_interactive_product_safety_policy(self) -> None:
        config = apply_runtime_profile(
            load_config(PROJECT_ROOT / "configs" / "pipeline.json"),
            "product-v7-safe",
        )

        recognition = config["recognition"]
        rendering = config["rendering"]
        verification = config["render_verification"]
        self.assertTrue(
            recognition["defer_photo_visual_handwriting_check_when_review_locked"]
        )
        self.assertTrue(rendering["enforce_end_to_end_deadline"])
        self.assertEqual(rendering["post_render_latency_reserve_seconds"], 0.20)
        self.assertEqual(rendering["edge_minimum_viable_attempt_seconds"], 1.20)
        self.assertTrue(verification["parallel_source_dual_ocr"])
        self.assertEqual(verification["source_dual_ocr_timeout_seconds"], 3.0)
        self.assertEqual(
            verification["source_dual_ocr_worker_recycle_after_requests"], 1
        )
        self.assertTrue(
            verification["source_dual_ocr_worker_recycle_after_timeout"]
        )
        self.assertFalse(verification["online_render_readback_enabled"])
        self.assertEqual(verification["render_readback_max_dimension_px"], 640)
        self.assertNotIn("source_dual_ocr_max_dimension_px", verification)

    def test_standard_profile_does_not_change_pipeline_configuration(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        baseline = copy.deepcopy(config)

        self.assertIs(apply_runtime_profile(config, "standard"), config)
        self.assertEqual(config, baseline)

    def test_cli_exposes_reproducible_product_runtime_profile(self) -> None:
        args = build_parser().parse_args(
            [
                "batch",
                "--runtime-profile",
                "product-v7-safe",
                "--limit",
                "0",
            ]
        )

        self.assertEqual(args.runtime_profile, "product-v7-safe")

    def test_unknown_runtime_profile_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            apply_runtime_profile({}, "unknown")

    def test_only_exact_gkp_source_tokens_enable_product_authentication(self) -> None:
        self.assertEqual(
            authenticated_source_sample_id("GKP-PHY-0047"),
            "GKP-PHY-0047",
        )
        self.assertIsNone(authenticated_source_sample_id("TZIP-PHYS-T01-Q01"))
        self.assertIsNone(authenticated_source_sample_id("GKP-PHY-47"))
        self.assertIsNone(authenticated_source_sample_id("GKP-PHY-0047.png"))


if __name__ == "__main__":
    unittest.main()
