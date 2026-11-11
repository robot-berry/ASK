from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "storage_schema" / "question_record.schema.json"
EXAMPLE_PATH = ROOT / "storage_schema" / "question_record.example.json"


class QuestionRecordSchemaTests(unittest.TestCase):
    def test_example_obeys_required_storage_contract(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(set(example), set(schema["required"]))
        self.assertFalse(schema["additionalProperties"])
        properties = schema["properties"]
        self.assertIn(example["学科类型"], properties["学科类型"]["enum"])
        self.assertIn(example["题目类型"], properties["题目类型"]["enum"])
        self.assertIn(example["解析等级"], properties["解析等级"]["enum"])
        self.assertRegex(
            example["图片上传时间"],
            re.compile(properties["图片上传时间"]["pattern"]),
        )
        datetime.strptime(example["图片上传时间"], "%Y-%m-%d %H:%M:%S")
        self.assertEqual(
            len(example["涉及知识点"]),
            len(set(example["涉及知识点"])),
        )
        self.assertRegex(
            example["设备SN码"],
            re.compile(properties["设备SN码"]["pattern"]),
        )


if __name__ == "__main__":
    unittest.main()
