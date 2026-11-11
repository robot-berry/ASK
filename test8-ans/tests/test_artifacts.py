from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from test8_ans.artifacts import export_comparison_artifacts


class ArtifactExportTests(unittest.TestCase):
    def test_exports_two_sides_and_latency_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch"
            batch.mkdir()
            run = root / "run"
            guidance = run / "images_800px"
            guidance.mkdir(parents=True)
            for tier in ("l1", "l2", "l3", "l4", "l5"):
                (guidance / f"{tier}.png").write_bytes(b"png")
            screenshot = run / "model_answer_screenshot"
            screenshot.mkdir()
            (screenshot / "model_output.png").write_bytes(b"png")
            source_image = root / "source.png"
            source_image.write_bytes(b"png")
            (run / "input.json").write_text(
                '{"source_image": "' + str(source_image).replace("\\", "\\\\") + '"}',
                encoding="utf-8",
            )
            report = export_comparison_artifacts(
                batch,
                [
                    {
                        "id": "q-1",
                        "subject": "math",
                        "status": "completed",
                        "quality_gate_pass": True,
                        "latency_pass": True,
                        "latency": {
                            "model_ms": 10_000.0,
                            "files_ready_with_png_ms": 10_150.0,
                        },
                        "run_dir": str(run),
                    }
                ],
            )
            self.assertEqual(report["guidance_complete_count"], 1)
            self.assertEqual(report["model_answer_screenshot_count"], 1)
            self.assertEqual(report["source_question_image_count"], 1)
            self.assertTrue((batch / "comparison_artifacts" / "overview.html").is_file())
            self.assertTrue(
                (batch / "comparison_artifacts" / "latency_comparison.csv").is_file()
            )
            overview = (batch / "comparison_artifacts" / "overview.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("审核反馈", overview)
            self.assertIn("export-feedback-json", overview)
            self.assertIn("localStorage", overview)


if __name__ == "__main__":
    unittest.main()
