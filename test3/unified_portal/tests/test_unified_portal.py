from __future__ import annotations

import asyncio
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
sys.path.insert(0, str(SOURCE_DIR))

import unified_portal_app as portal  # noqa: E402


class UnifiedPortalContractTests(unittest.TestCase):
    def completed_payload(self) -> dict:
        return {
            "job_id": "20260814T120000_0123456789abcdef0123456789abcdef",
            "status": "completed",
            "created_at": "2026-08-14T12:00:00+08:00",
            "updated_at": "2026-08-14T12:01:00+08:00",
            "images": [
                {"filename": filename, "url": f"/jobs/private/{filename}"}
                for filename in reversed(portal.DEVICE_IMAGE_NAMES)
            ],
        }

    def test_completed_job_is_ordered_and_rewritten_to_same_origin(self) -> None:
        public = portal.sanitize_job("math", self.completed_payload())
        self.assertEqual(
            [item["filename"] for item in public["images"]],
            list(portal.DEVICE_IMAGE_NAMES),
        )
        for item in public["images"]:
            self.assertTrue(item["url"].startswith("/api/math/jobs/"))
            self.assertNotIn("127.0.0.1", item["url"])
            self.assertNotIn(":8020", item["url"])

    def test_only_the_fixed_four_png_names_are_accepted(self) -> None:
        payload = self.completed_payload()
        payload["images"].append({"filename": "05_answer.png"})
        with self.assertRaises(portal.UpstreamFailure):
            portal.sanitize_job("physics", payload)
        with self.assertRaises(HTTPException) as context:
            portal.validate_result_filename("05_answer.png")
        self.assertEqual(context.exception.status_code, 404)

    def test_subject_and_upload_limits_are_enforced(self) -> None:
        with self.assertRaises(HTTPException) as context:
            portal.backend_for_subject("chemistry")
        self.assertEqual(context.exception.status_code, 404)
        with self.assertRaises(HTTPException) as context:
            portal.validate_upload(b"x" * (portal.MAX_UPLOAD_BYTES + 1), "image/jpeg")
        self.assertEqual(context.exception.status_code, 413)
        with self.assertRaises(HTTPException) as context:
            portal.validate_upload(b"not-an-image", "text/plain")
        self.assertEqual(context.exception.status_code, 415)
        with self.assertRaises(HTTPException) as context:
            portal.validate_upload(b"image", "image/jpeg\r\nX-Injected: yes")
        self.assertEqual(context.exception.status_code, 415)

    def test_page_never_addresses_backend_ports(self) -> None:
        html = (SOURCE_DIR / "static" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("8010", html)
        self.assertNotIn("8020", html)
        self.assertNotIn("127.0.0.1", html)
        self.assertIn("/api/${encodeURIComponent(subject)}/jobs", html)
        self.assertIn("数学覆盖小学、初中、高中", html)
        self.assertIn("物理覆盖小学科学中的物理内容、初中、高中", html)
        self.assertIn("当前主要面向完整的非选择题", html)
        self.assertIn("计算题、解答题、证明题、实验探究题、简答题和综合应用题", html)
        self.assertIn("选择题、填空题、判断题暂不承诺答案隔离", html)
        self.assertIn("默认保留照片中的全部笔迹", html)
        self.assertNotIn("只支持高中", html)

        readme = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("数学适用于小学、初中、高中", readme)
        self.assertIn("物理适用于小学科学中的物理内容、初中、高中", readme)
        self.assertIn("当前主要面向完整的非选择题", readme)
        self.assertIn("选择题、填空题、判断题暂不承诺答案隔离", readme)
        self.assertIn("不应理解为已经覆盖所有题型", readme)

    def test_health_aggregates_both_subjects(self) -> None:
        states = {
            "physics": {"available": True, "status": "ok"},
            "math": {"available": False, "status": "unavailable"},
        }
        with patch.object(
            portal,
            "_subject_health_sync",
            side_effect=lambda subject: states[subject],
        ):
            response = asyncio.run(portal.health())
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body)
        self.assertEqual(body["status"], "degraded")
        self.assertTrue(body["subjects"]["physics"]["available"])
        self.assertFalse(body["subjects"]["math"]["available"])

    def test_public_json_is_private_and_not_cached(self) -> None:
        response = portal.public_json({"status": "queued"}, status_code=202)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

    def test_result_png_must_decode_and_match_test3_geometry(self) -> None:
        buffer = BytesIO()
        Image.new("L", (800, 120), 255).save(buffer, format="PNG")
        data = buffer.getvalue()
        portal.validate_device_png(data)
        with self.assertRaises(portal.UpstreamFailure):
            portal.validate_device_png(data[:40])
        wrong = BytesIO()
        Image.new("RGB", (800, 120), "white").save(wrong, format="PNG")
        with self.assertRaises(portal.UpstreamFailure):
            portal.validate_device_png(wrong.getvalue())


if __name__ == "__main__":
    unittest.main()
