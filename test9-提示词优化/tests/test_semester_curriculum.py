import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import semester_curriculum as sc
from compact_guidance import curriculum_scope_instruction

class SemesterTests(unittest.TestCase):
    def test_all_twelve_terms_have_distinct_subject_prompts(self):
        terms=sc.catalog()
        self.assertEqual(len(terms),12)
        for subject in ("math","physics"):
            revisions=set()
            previous=set()
            for item in terms:
                selected=sc.snapshot(subject,item["id"])
                revisions.add(selected["revision"])
                self.assertIn(item["label"],selected["prompt"])
                allowed=set(item["subjects"][subject]["allowed"])
                self.assertTrue(previous <= allowed)
                self.assertFalse(allowed & set(item["subjects"][subject]["not_yet_learned"]))
                previous=allowed
            self.assertEqual(len(revisions),12)

    def test_explicit_selection_overrides_ocr_stage_both_directions(self):
        for term, stage in (("g7_s1","junior_high"),("g12_s2","senior_high")):
            selected=sc.snapshot("math",term)
            scope=sc.apply_scope({"stage":"primary","subject":"mathematics"},selected)
            self.assertEqual(scope["stage"],stage)
            self.assertIn(selected["label"],curriculum_scope_instruction(scope))
        self.assertIn("本学期允许的数学工具上限",sc.snapshot("physics","g8_s1")["prompt"])

    def test_invalid_term_rejected_and_empty_preserves_legacy(self):
        for value in ("g6_s1","g7_s3","../../etc","g13_s1"):
            with self.assertRaises(ValueError):sc.snapshot("math",value)
        self.assertIsNone(sc.snapshot("math",""))
        inferred={"stage":"primary"}
        self.assertEqual(sc.apply_scope(inferred,None),inferred)

    def test_snapshot_is_not_changed_by_later_file_edits(self):
        original=sc.snapshot("math","g8_s1")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"curriculum_terms.json").write_text(json.dumps(sc.catalog()),encoding="utf-8")
            folder=root/"curriculum_prompts/math";folder.mkdir(parents=True)
            (folder/"g8_s1.md").write_text("changed",encoding="utf-8")
            with patch.object(sc,"ROOT",root):updated=sc.snapshot("math","g8_s1")
        self.assertNotEqual(original["revision"],updated["revision"])
        self.assertEqual(original["revision"],hashlib.sha256(original["prompt"].encode()).hexdigest())
