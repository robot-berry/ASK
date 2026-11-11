import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import curriculum_profile
from compact_guidance import curriculum_scope_instruction

class CurriculumProfileTests(unittest.TestCase):
    def test_subject_context_reaches_boundary(self):
        data = json.loads((curriculum_profile.ROOT / "curriculum_profile.json").read_text(encoding="utf-8"))
        data["math"]["learned_knowledge"] = ["一元一次方程"]
        data["physics"]["not_yet_learned"] = ["动量"]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "curriculum_profile.json").write_text(json.dumps(data), encoding="utf-8")
            with patch.object(curriculum_profile, "ROOT", Path(tmp)):
                math = curriculum_scope_instruction({"stage":"junior_high", "subject":"mathematics"})
                physics = curriculum_scope_instruction({"stage":"senior_high", "subject":"physics"})
        self.assertIn("一元一次方程", math)
        self.assertNotIn("一元一次方程", physics)
        self.assertIn('"not_yet_learned": ["动量"]', physics)
        self.assertIn("教材归属", math)

    def test_invalid_profile_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "curriculum_profile.json").write_text('{"math": {"grade": 8}}', encoding="utf-8")
            with patch.object(curriculum_profile, "ROOT", Path(tmp)):
                with self.assertRaises(ValueError):
                    curriculum_profile.instruction("math")
