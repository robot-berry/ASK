from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PORTAL = ROOT / "portal" / "source" / "test4_portal_app.py"
TOKEN = "test4-unit-token-0123456789abcdef"
os.environ["TEST4_ACCESS_TOKEN"] = TOKEN
os.environ["TEST4_COOKIE_SECURE"] = "0"
_STATE_DIR = tempfile.TemporaryDirectory(prefix="test4-portal-tests-")
_STATE_DB = str(Path(_STATE_DIR.name) / "gateway.sqlite3")
# The production launcher uses the TEST6 name while compatibility tests also
# exercise the legacy TEST4 name.  Pin both to the same disposable database so
# an inherited launcher variable can never redirect tests into live state.
os.environ["TEST6_PORTAL_STATE_DB"] = _STATE_DB
os.environ["TEST4_PORTAL_STATE_DB"] = _STATE_DB
spec = importlib.util.spec_from_file_location("test4_portal_test_module", PORTAL)
assert spec and spec.loader
portal = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = portal
spec.loader.exec_module(portal)


def tearDownModule() -> None:
    # addModuleCleanup uses a process-global stack and can run after an earlier
    # discovered module.  A module teardown keeps this directory alive until
    # the portal tests themselves have finished.
    _STATE_DIR.cleanup()


def state(status: str = "queued") -> dict:
    payload = {
        "job_id": "20260815T120000_0123456789abcdef0123456789abcdef",
        "status": status,
        "created_at": "2026-08-15T12:00:00+00:00",
        "updated_at": "2026-08-15T12:00:01+00:00",
        "elapsed_seconds": 1.0,
        "error": None,
        "images": [],
    }
    if status == "completed":
        payload["images"] = [
            {
                "filename": filename,
                "width": 800,
                "height": 1000 + index,
                "mode": "L",
                "format": "PNG",
                "bytes": 100 + index,
                "sha256": f"{index:064x}",
                "display_width_cm": 9.0,
            }
            for index, filename in enumerate(portal.DEVICE_IMAGE_NAMES, start=1)
        ]
        payload["analysis_audit"] = {
            "source": "complete_model",
            "model_id": "qwen3.8-max",
            "knowledge_points": ["频数分布直方图"],
            "base_formulas": [
                {"formula": "P(A)=m/n", "condition": "古典概型"}
            ],
            "parts": [
                {
                    "part": "（1）",
                    "methods": [
                        {
                            "method_id": "M1",
                            "method_name": "读图分析",
                            "steps": ["由总人数与概率的关系可确定对应组频数"],
                            "pitfall": "不要混淆频数与频率",
                        }
                    ],
                }
            ],
        }
    return payload


class PortalTests(unittest.TestCase):
    def setUp(self) -> None:
        connection = portal._open_idempotency_db()
        try:
            connection.execute("DELETE FROM idempotency_jobs")
        finally:
            connection.close()
        self.client = TestClient(portal.app)

    def test_bearer_is_required_for_hardware_api(self) -> None:
        self.assertEqual(self.client.get("/api/v1/capabilities").status_code, 401)
        response = self.client.get(
            "/api/v1/capabilities",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["api_version"], "v1")
        self.assertEqual([item["id"] for item in body["subjects"]], ["physics", "math"])
        self.assertEqual(len(body["results"]), 6)
        self.assertEqual(
            body["upload"]["idempotency"]["header"],
            "Idempotency-Key",
        )

    def test_individual_hardware_token_hash_is_supported(self) -> None:
        device_token = "camera-07-random-secret-0123456789"
        digest = hashlib.sha256(device_token.encode("utf-8")).hexdigest()
        with patch.object(portal, "DEVICE_TOKEN_HASHES", (digest,)):
            response = self.client.get(
                "/api/v1/capabilities",
                headers={"Authorization": f"Bearer {device_token}"},
            )
        self.assertEqual(response.status_code, 200)

    def test_backend_override_can_only_target_loopback(self) -> None:
        with patch.dict(os.environ, {"TEST4_TEST_BACKEND": "http://127.0.0.1:8110"}):
            self.assertEqual(
                portal._loopback_backend("TEST4_TEST_BACKEND", "http://127.0.0.1:1"),
                "http://127.0.0.1:8110",
            )
        with patch.dict(os.environ, {"TEST4_TEST_BACKEND": "https://example.com:443"}):
            with self.assertRaises(RuntimeError):
                portal._loopback_backend("TEST4_TEST_BACKEND", "http://127.0.0.1:1")

    def test_browser_login_uses_derived_http_only_cookie(self) -> None:
        bad = self.client.post("/api/v1/auth/login", json={"token": "wrong"})
        self.assertEqual(bad.status_code, 401)
        good = self.client.post("/api/v1/auth/login", json={"token": TOKEN})
        self.assertEqual(good.status_code, 200)
        cookie = self.client.cookies.get(portal.COOKIE_NAME)
        self.assertTrue(cookie)
        self.assertNotEqual(cookie, TOKEN)
        self.assertTrue(self.client.get("/api/v1/capabilities").json()["api_version"])
        # An explicit invalid hardware credential must not silently fall back
        # to a valid browser cookie.
        denied = self.client.get(
            "/api/v1/capabilities",
            headers={"Authorization": "Bearer wrong-device-token"},
        )
        self.assertEqual(denied.status_code, 401)

    def test_expired_browser_session_is_rejected_server_side(self) -> None:
        expired = portal._encode_session(1, "0" * 32)
        self.client.cookies.set(portal.COOKIE_NAME, expired)
        self.assertEqual(self.client.get("/api/v1/capabilities").status_code, 401)

    def test_raw_hardware_upload_returns_poll_contract(self) -> None:
        with patch.object(portal, "_submit_job_sync", return_value=state()) as submit:
            response = self.client.post(
                "/api/v1/math/jobs/raw",
                content=b"encoded-jpeg",
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Type": "image/jpeg",
                    "X-Device-SN": "ASK-CAMERA-007",
                },
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.headers["retry-after"], "2")
        body = response.json()
        self.assertEqual(response.headers["location"], body["poll_url"])
        self.assertEqual(body["subject"], "math")
        self.assertEqual(body["status"], "queued")
        args = submit.call_args.args
        primary = portal._principal_id("primary", TOKEN)
        self.assertEqual(
            args[3],
            portal._request_device_principal(primary, "ASK-CAMERA-007"),
        )
        self.assertEqual(args[4], "ASK-CAMERA-007")

    def test_invalid_device_sn_is_rejected_before_upstream_submission(self) -> None:
        with patch.object(portal, "_submit_job_sync", return_value=state()) as submit:
            response = self.client.post(
                "/api/v1/math/jobs/raw",
                content=b"encoded-jpeg",
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Type": "image/jpeg",
                    "X-Device-SN": "bad sn with spaces",
                },
            )
        self.assertEqual(response.status_code, 400)
        submit.assert_not_called()

    def test_phone_multipart_upload_uses_the_same_contract(self) -> None:
        with patch.object(portal, "_submit_job_sync", return_value=state()):
            response = self.client.post(
                "/api/v1/physics/jobs",
                files={"image": ("camera.jpg", b"encoded-jpeg", "image/jpeg")},
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["subject"], "physics")
        self.assertEqual(response.headers["retry-after"], "2")

    def test_idempotency_replays_same_job_across_raw_and_multipart(self) -> None:
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "image/jpeg",
            "Idempotency-Key": "camera07-20260815-000001",
        }
        with patch.object(portal, "_submit_job_sync", return_value=state()) as submit:
            first = self.client.post(
                "/api/v1/math/jobs/raw",
                content=b"same-encoded-image",
                headers=headers,
            )
            second = self.client.post(
                "/api/v1/math/jobs",
                files={
                    "image": (
                        "camera.jpg",
                        b"same-encoded-image",
                        "image/jpeg",
                    )
                },
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Idempotency-Key": "camera07-20260815-000001",
                },
            )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(first.json()["job_id"], second.json()["job_id"])
        self.assertEqual(first.headers["idempotency-replayed"], "false")
        self.assertEqual(second.headers["idempotency-replayed"], "true")
        self.assertEqual(first.headers["location"], second.headers["location"])
        self.assertEqual(submit.call_count, 1)

        connection = portal._open_idempotency_db()
        try:
            self.assertEqual(
                connection.execute("PRAGMA journal_mode").fetchone()[0].lower(),
                "wal",
            )
            row = connection.execute(
                "SELECT principal_id, key_sha256, image_sha256 FROM idempotency_jobs"
            ).fetchone()
        finally:
            connection.close()
        self.assertIsNotNone(row)
        self.assertNotEqual(row["principal_id"], TOKEN)
        self.assertNotEqual(row["key_sha256"], "camera07-20260815-000001")

    def test_idempotency_rejects_same_key_for_another_image(self) -> None:
        base_headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "image/jpeg",
            "Idempotency-Key": "camera07-20260815-000002",
        }
        with patch.object(portal, "_submit_job_sync", return_value=state()) as submit:
            first = self.client.post(
                "/api/v1/physics/jobs/raw",
                content=b"image-one",
                headers=base_headers,
            )
            conflict = self.client.post(
                "/api/v1/physics/jobs/raw",
                content=b"image-two",
                headers=base_headers,
            )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(submit.call_count, 1)

    def test_idempotency_key_is_strictly_validated(self) -> None:
        invalid_keys = (
            "",
            "short",
            "contains space",
            "contains/slash",
            "x" * 129,
        )
        with patch.object(portal, "_submit_job_sync", return_value=state()) as submit:
            for key in invalid_keys:
                with self.subTest(key=key[:20]):
                    response = self.client.post(
                        "/api/v1/math/jobs/raw",
                        content=b"image",
                        headers={
                            "Authorization": f"Bearer {TOKEN}",
                            "Content-Type": "image/jpeg",
                            "Idempotency-Key": key,
                        },
                    )
                    self.assertEqual(response.status_code, 400)
            duplicate = self.client.post(
                "/api/v1/math/jobs/raw",
                content=b"image",
                headers=[
                    ("Authorization", f"Bearer {TOKEN}"),
                    ("Content-Type", "image/jpeg"),
                    ("Idempotency-Key", "valid-key-000001"),
                    ("Idempotency-Key", "valid-key-000002"),
                ],
            )
            self.assertEqual(duplicate.status_code, 400)
        submit.assert_not_called()

    def test_idempotency_key_accepts_exact_length_boundaries(self) -> None:
        with patch.object(portal, "_submit_job_sync", return_value=state()) as submit:
            responses = [
                self.client.post(
                    "/api/v1/math/jobs/raw",
                    content=f"image-{index}".encode(),
                    headers={
                        "Authorization": f"Bearer {TOKEN}",
                        "Content-Type": "image/jpeg",
                        "Idempotency-Key": key,
                    },
                )
                for index, key in enumerate(("a1234567", "z" * 128))
            ]
        self.assertEqual([response.status_code for response in responses], [202, 202])
        self.assertEqual(submit.call_count, 2)

    def test_idempotency_scope_includes_principal_and_subject(self) -> None:
        device_token = "camera-08-random-secret-0123456789"
        digest = hashlib.sha256(device_token.encode("utf-8")).hexdigest()
        key = "shared-across-principals-0001"
        with (
            patch.object(portal, "DEVICE_TOKEN_HASHES", (digest,)),
            patch.object(portal, "_submit_job_sync", return_value=state()) as submit,
        ):
            responses = [
                self.client.post(
                    "/api/v1/math/jobs/raw",
                    content=b"image",
                    headers={
                        "Authorization": f"Bearer {TOKEN}",
                        "Content-Type": "image/jpeg",
                        "Idempotency-Key": key,
                    },
                ),
                self.client.post(
                    "/api/v1/math/jobs/raw",
                    content=b"image",
                    headers={
                        "Authorization": f"Bearer {device_token}",
                        "Content-Type": "image/jpeg",
                        "Idempotency-Key": key,
                    },
                ),
                self.client.post(
                    "/api/v1/physics/jobs/raw",
                    content=b"image",
                    headers={
                        "Authorization": f"Bearer {TOKEN}",
                        "Content-Type": "image/jpeg",
                        "Idempotency-Key": key,
                    },
                ),
            ]
        self.assertEqual([response.status_code for response in responses], [202] * 3)
        self.assertEqual(submit.call_count, 3)

    def test_concurrent_same_key_has_one_upstream_submission(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow_submit(*_args) -> dict:
            started.set()
            if not release.wait(5):
                raise AssertionError("test did not release the upstream request")
            return state()

        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "image/jpeg",
            "Idempotency-Key": "camera09-concurrent-000001",
        }

        def post_once():
            with TestClient(portal.app) as client:
                return client.post(
                    "/api/v1/math/jobs/raw",
                    content=b"same-image",
                    headers=headers,
                )

        with patch.object(portal, "_submit_job_sync", side_effect=slow_submit) as submit:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first_future = executor.submit(post_once)
                self.assertTrue(started.wait(2))
                second_future = executor.submit(post_once)
                time.sleep(0.15)
                release.set()
                first = first_future.result(timeout=5)
                second = second_future.result(timeout=5)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(first.json()["job_id"], second.json()["job_id"])
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(
            {first.headers["idempotency-replayed"], second.headers["idempotency-replayed"]},
            {"false", "true"},
        )

    def test_upstream_failure_releases_key_for_retry(self) -> None:
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "image/jpeg",
            "Idempotency-Key": "camera10-retry-after-failure",
        }
        with patch.object(
            portal,
            "_submit_job_sync",
            side_effect=[portal.UpstreamFailure(503, "unavailable"), state()],
        ) as submit:
            failed = self.client.post(
                "/api/v1/physics/jobs/raw",
                content=b"same-image",
                headers=headers,
            )
            retried = self.client.post(
                "/api/v1/physics/jobs/raw",
                content=b"same-image",
                headers=headers,
            )
        self.assertEqual(failed.status_code, 503)
        self.assertEqual(retried.status_code, 202)
        self.assertEqual(submit.call_count, 2)

    def test_upstream_queue_full_preserves_retry_after(self) -> None:
        with patch.object(
            portal,
            "_submit_job_sync",
            side_effect=portal.UpstreamFailure(429, "queue full"),
        ):
            response = self.client.post(
                "/api/v1/math/jobs/raw",
                content=b"image",
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Type": "image/jpeg",
                },
            )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "2")

    def test_unsupported_hardware_media_type_is_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/math/jobs/raw",
            content=b"heic",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "image/heic",
            },
        )
        self.assertEqual(response.status_code, 415)

    def test_webp_raw_upload_keeps_webp_filename_for_backend(self) -> None:
        body, _ = portal._multipart_upload(b"webp", "image/webp")
        self.assertIn(b'filename="camera.webp"', body)
        self.assertIn(b"Content-Type: image/webp", body)

    def test_completed_job_has_exact_order_and_safe_metadata(self) -> None:
        job_id = state()["job_id"]
        with patch.object(portal, "_fetch_job_sync", return_value=state("completed")):
            response = self.client.get(
                f"/api/v1/physics/jobs/{job_id}",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["result_contract"]["complete"])
        self.assertEqual(body["result_contract"]["name"], "test6-six-png")
        self.assertEqual(
            [item["filename"] for item in body["images"]],
            list(portal.DEVICE_IMAGE_NAMES),
        )
        self.assertEqual(
            [item["sequence"] for item in body["images"]],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertTrue(all(item["width"] == 800 for item in body["images"]))
        self.assertEqual(body["analysis_audit"]["source"], "complete_model")
        self.assertEqual(
            body["analysis_audit"]["parts"][0]["methods"][0]["steps"],
            ["由总人数与概率的关系可确定对应组频数"],
        )
        self.assertIn("强校验", body["analysis_audit"]["notice"])

    def test_safe_latency_milestones_are_exposed_in_seconds(self) -> None:
        completed = state("completed")
        completed["elapsed_seconds"] = 18.625
        completed["streaming"] = {
            "first_image_seconds": 0.14,
            "first_image_from_created_seconds": 0.64,
            "first_image_refined_seconds": 2.34,
            "first_image_refinement_pending": False,
            "provider_prefix_ms": 812.5,
            "first_guidance_from_created_seconds": 4.25,
            "model_first_content_ms": 500,
            "guidance_tiers": {
                "elite": {"published": True, "first_published_from_created_seconds": 4.25,
                          "private_source_path": "D:/tier-private.png"},
                "upper": {"published": False, "error": "private diagnostic"},
                "unknown": {"published": True},
            },
            "private_source_path": "D:/must-not-leak.png",
        }
        job_id = completed["job_id"]
        with patch.object(portal, "_fetch_job_sync", return_value=completed):
            response = self.client.get(
                f"/api/v1/math/jobs/{job_id}",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["timing"],
            {
                "queue_wait_seconds": 0.5,
                "first_image_processing_seconds": 0.14,
                "first_image_from_created_seconds": 0.64,
                "first_guidance_from_created_seconds": 4.25,
                "model_first_content_seconds": 0.5,
                "guidance_tiers": {"elite": {"first_published_from_created_seconds": 4.25}},
                "first_image_refined_processing_seconds": 2.34,
                "provider_prefix_seconds": 0.812,
                "processing_seconds": 18.625,
                "first_image_refinement_pending": False,
            },
        )
        self.assertNotIn("private_source_path", str(body))

    def test_test_page_contains_live_latency_panel(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.text
        self.assertIn('id="latencyPanel"', page)
        self.assertIn('id="latencyFirst"', page)
        self.assertIn('id="latencyGuidance"', page)
        self.assertIn('id="latencyRefined"', page)
        self.assertIn('id="latencyTotal"', page)
        self.assertIn("网页实测包含上传", page)

    def test_test_page_displays_unambiguous_model_provenance(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.text
        self.assertIn('id="sourcePanel"', page)
        self.assertIn("解析来源：大模型回答已采用", page)
        self.assertIn("解析来源：本地兜底，未采用完整模型回答", page)
        self.assertIn("audit.source", page)

    def test_invalid_analysis_audit_is_rejected(self) -> None:
        broken = state("completed")
        broken["analysis_audit"]["source"] = "untrusted"
        job_id = broken["job_id"]
        with patch.object(portal, "_fetch_job_sync", return_value=broken):
            response = self.client.get(
                f"/api/v1/math/jobs/{job_id}",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
        self.assertEqual(response.status_code, 502)

    def test_model_five_tiers_are_preserved_for_browser_inspection(self) -> None:
        completed = state("completed")
        base_part = completed["analysis_audit"]["parts"][0]
        completed["analysis_audit"]["tiers"] = [
            {
                "tier_key": key,
                "title": title,
                "knowledge_points": ["频数分布"] if key in {"upper", "novice"} else [],
                "base_formulas": [],
                "parts": [] if key == "upper" else [base_part],
                "pattern_summary": "" if key != "novice" else "先读图再建立关系",
            }
            for key, title in (
                ("elite", "学霸解析"),
                ("upper", "中上解析"),
                ("middle", "中等解析"),
                ("lower", "中下解析"),
                ("novice", "学渣解析"),
            )
        ]
        job_id = completed["job_id"]
        with patch.object(portal, "_fetch_job_sync", return_value=completed):
            response = self.client.get(
                f"/api/v1/math/jobs/{job_id}",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )

        self.assertEqual(response.status_code, 200)
        audit = response.json()["analysis_audit"]
        self.assertEqual(
            [tier["tier_key"] for tier in audit["tiers"]],
            ["elite", "upper", "middle", "lower", "novice"],
        )
        self.assertIn("五档", audit["notice"])
        self.assertEqual(audit["tiers"][4]["pattern_summary"], "先读图再建立关系")

    def test_public_analysis_text_never_exposes_latex_source_markup(self) -> None:
        completed = state("completed")
        completed["analysis_audit"]["base_formulas"] = [
            {
                "formula": r"$P(A)=\frac{m}{n}$",
                "condition": r"当 $n\geq 1$ 时",
            }
        ]
        completed["analysis_audit"]["parts"][0]["methods"][0]["steps"] = [
            r"由 $P(A)=\frac{m}{n}$ 建立关系"
        ]
        job_id = completed["job_id"]
        with patch.object(portal, "_fetch_job_sync", return_value=completed):
            response = self.client.get(
                f"/api/v1/math/jobs/{job_id}",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )

        self.assertEqual(response.status_code, 200)
        audit = response.json()["analysis_audit"]
        serialized = str(audit)
        self.assertNotIn("\\frac", serialized)
        self.assertNotIn("$", serialized)
        self.assertEqual(audit["base_formulas"][0]["formula"], "P(A)=m/n")
        self.assertEqual(audit["base_formulas"][0]["condition"], "当 n≥ 1 时")
        self.assertEqual(
            audit["parts"][0]["methods"][0]["steps"],
            ["由 P(A)=m/n 建立关系"],
        )

    def test_rendering_job_exposes_only_the_generated_ordered_prefix(self) -> None:
        partial = state("rendering")
        partial["images"] = state("completed")["images"][:2]
        job_id = partial["job_id"]
        with patch.object(portal, "_fetch_job_sync", return_value=partial):
            response = self.client.get(
                f"/api/v1/math/jobs/{job_id}",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["result_contract"]["complete"])
        self.assertEqual(body["result_contract"]["available_images"], 2)
        self.assertEqual(
            [item["filename"] for item in body["images"]],
            list(portal.DEVICE_IMAGE_NAMES[:2]),
        )
        self.assertNotIn("path", str(body))

    def test_completed_job_with_invalid_png_metadata_is_rejected(self) -> None:
        broken = state("completed")
        broken["images"][0]["width"] = 799
        job_id = broken["job_id"]
        with patch.object(portal, "_fetch_job_sync", return_value=broken):
            response = self.client.get(
                f"/api/v1/math/jobs/{job_id}",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
        self.assertEqual(response.status_code, 502)

    def test_result_image_is_protected(self) -> None:
        image = Image.new("L", (800, 12), 255)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        job_id = state()["job_id"]
        url = f"/api/v1/math/jobs/{job_id}/images/{portal.DEVICE_IMAGE_NAMES[0]}"
        self.assertEqual(self.client.get(url).status_code, 401)
        with patch.object(portal, "_fetch_png_sync", return_value=buffer.getvalue()):
            response = self.client.get(
                url,
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")


if __name__ == "__main__":
    unittest.main()
