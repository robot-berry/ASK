from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StartScriptTests(unittest.TestCase):
    def test_documented_unittest_discovery_root_is_importable(self) -> None:
        tests_package = PROJECT_ROOT / "tests" / "__init__.py"
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertTrue(tests_package.is_file())
        self.assertIn(
            r".\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v",
            readme,
        )

    def test_product_restart_switch_forces_review_service_reload(self) -> None:
        product = (PROJECT_ROOT / "scripts" / "start_product_ocr.ps1").read_text(
            encoding="utf-8"
        )
        review = (PROJECT_ROOT / "scripts" / "start_gold_review.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("[switch]$Restart", product)
        self.assertIn("-Restart:$Restart", product)
        self.assertIn("[switch]$Restart", review)
        self.assertIn("-not $Restart -and", review)

    def test_self_built_check_defaults_to_concise_status_with_detailed_opt_in(self) -> None:
        script = (PROJECT_ROOT / "scripts" / "run_self_built.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("[switch]$Detailed", script)
        self.assertIn("test7-OCR 自建数据状态", script)
        self.assertIn("真实采集：", script)
        self.assertIn("run_self_built.ps1 -Detailed", script)


if __name__ == "__main__":
    unittest.main()
