from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from test7_ocr.resource_guard import probe_batch_resource_conflicts


class _HealthResponse(io.BytesIO):
    status = 200


class ResourceGuardTests(unittest.TestCase):
    def test_rejects_non_loopback_health_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            probe_batch_resource_conflicts(
                {
                    "enabled": True,
                    "conflicting_services": ["http://example.com:8260/health"],
                }
            )

    def test_marks_responding_local_service_as_conflict(self) -> None:
        response = _HealthResponse(
            json.dumps(
                {"status": "ok", "ready": True, "engine": "other-formula-engine"}
            ).encode("utf-8")
        )
        with patch("test7_ocr.resource_guard.urlopen", return_value=response):
            result = probe_batch_resource_conflicts(
                {
                    "enabled": True,
                    "conflicting_services": [
                        {
                            "name": "other OCR",
                            "health_url": "http://127.0.0.1:8260/health",
                        }
                    ],
                }
            )
        self.assertEqual(result["status"], "CONFLICT")
        self.assertEqual(result["conflicts"][0]["engine"], "other-formula-engine")

    def test_connection_failure_means_no_active_conflict(self) -> None:
        from urllib.error import URLError

        with patch("test7_ocr.resource_guard.urlopen", side_effect=URLError("refused")):
            result = probe_batch_resource_conflicts(
                {
                    "enabled": True,
                    "conflicting_services": ["http://localhost:8260/health"],
                }
            )
        self.assertEqual(result["status"], "CLEAR")
        self.assertEqual(result["unavailable"][0]["reason"], "URLError")


if __name__ == "__main__":
    unittest.main()
