from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
MATH_SOURCE = ROOT / "math" / "source"
for candidate in (ROOT, MATH_SOURCE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from local_ocr_content import (  # noqa: E402
    find_ocr_text_table,
    strip_likely_handwritten_work,
)
from math_guidance_renderer import render_local_ocr_question_page  # noqa: E402


TABLE_QUESTION = (
    "5. 某射手在一次射击训练中共射击10次，成绩如下表：\n"
    "环数 7环 8环 9环 10环\n"
    "次数 1 4 3 2\n"
    "则本次射击训练成绩的中位数是\n"
    "A. 8环 B. 8.5环 C. 9环 D. 9.5环"
)


class LocalOcrContentTests(unittest.TestCase):
    def test_complete_rectangular_rows_become_a_text_table(self) -> None:
        table = find_ocr_text_table(TABLE_QUESTION.splitlines())

        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.start, 1)
        self.assertEqual(table.end, 3)
        self.assertEqual(table.column_count, 5)
        self.assertEqual(table.rows[1], ("次数", "1", "4", "3", "2"))

    def test_incomplete_table_falls_back_instead_of_inventing_cells(self) -> None:
        lines = ["成绩如下表：", "环数 7环 8环 9环", "次数 1 4"]
        self.assertIsNone(find_ocr_text_table(lines))

    def test_working_between_prompt_and_choices_is_removed(self) -> None:
        source = (
            "6. 若方程有两个不相等的实数根，则实数m的取值\n"
            "范围是 4m2-4m2-4（M-23>0 x2-2=0\n"
            "A. m>2 B. m≥2 C. m<2 D. m<2且m≠0"
        )
        cleaned, actions = strip_likely_handwritten_work(source)

        self.assertIn("范围是\nA.", cleaned)
        self.assertNotIn("M-23", cleaned)
        self.assertIn("remove_working_between_prompt_and_choices", actions)

    def test_renderer_uses_text_table_and_hides_public_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_crop = root / "crop.png"
            reference = root / "reference.png"
            destination = root / "question.png"
            image = Image.new("L", (500, 160), 255)
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 480, 140), outline=0, width=2)
            image.save(source_crop)
            image.save(reference)

            audit = render_local_ocr_question_page(
                ocr_text=TABLE_QUESTION,
                illustration_images=[source_crop],
                semantic_reference_image=reference,
                destination=destination,
            )

            self.assertTrue(destination.is_file())
            self.assertEqual(audit["text_table_count"], 1)
            self.assertTrue(audit["table_crop_suppressed"])
            self.assertEqual(audit["illustration_count"], 0)
            self.assertFalse(audit["semantic_reference_included"])


if __name__ == "__main__":
    unittest.main()
