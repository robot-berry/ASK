from contextlib import closing
import io
import unittest
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from test_portal import portal,TOKEN,state

class SemesterPortalTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(portal.app)
        self.headers={"Authorization":"Bearer "+TOKEN}
        out=io.BytesIO();Image.new("RGB",(20,20),"white").save(out,format="PNG")
        self.image=out.getvalue()
        with closing(portal._open_idempotency_db()) as db:
            db.execute("DELETE FROM idempotency_jobs")
    def tearDown(self):self.client.close()
    def test_catalog_auth_and_twelve_previews(self):
        self.assertEqual(self.client.get("/api/v1/curriculum-terms").status_code,401)
        response=self.client.get("/api/v1/curriculum-terms",headers=self.headers)
        self.assertEqual(response.status_code,200)
        terms=response.json()["terms"]
        self.assertEqual(len(terms),12)
        self.assertIn("初一上学期",terms[0]["subjects"]["math"]["prompt"])
    def test_selected_term_forwarded_and_invalid_rejected(self):
        with patch.object(portal,"_submit_job_sync",return_value=state()) as submit:
            response=self.client.post("/api/v1/math/jobs",headers={**self.headers,"X-Curriculum-Term":"g8_s2"},files={"image":("q.png",self.image,"image/png")})
            self.assertEqual(response.status_code,202)
            self.assertEqual(submit.call_args.args[-1],"g8_s2")
            submit.reset_mock()
            response=self.client.post("/api/v1/math/jobs",headers={**self.headers,"X-Curriculum-Term":"g8_s9"},files={"image":("q.png",self.image,"image/png")})
            self.assertEqual(response.status_code,400)
            submit.assert_not_called()
    def test_same_key_with_different_term_conflicts(self):
        headers={**self.headers,"Idempotency-Key":"semester-test-123"}
        with patch.object(portal,"_submit_job_sync",return_value=state()) as submit:
            for term,expected in (("g7_s1",202),("g7_s1",202),("g8_s1",409)):
                response=self.client.post("/api/v1/math/jobs",headers={**headers,"X-Curriculum-Term":term},files={"image":("q.png",self.image,"image/png")})
                self.assertEqual(response.status_code,expected)
            self.assertEqual(submit.call_count,1)
    def test_upstream_header_and_public_snapshot(self):
        with patch.object(portal,"_request_json",return_value=state()) as send:
            portal._submit_job_sync("physics",self.image,"image/png","device",None,"g11_s2")
            self.assertEqual(send.call_args.kwargs["headers"]["X-Curriculum-Term"],"g11_s2")
        payload=state();payload["curriculum_selection"]={"id":"g11_s2","label":"高二下学期","revision":"abc","prompt":"private"}
        public=portal.sanitize_job("physics",payload)
        self.assertEqual(public["curriculum_selection"]["label"],"高二下学期")
        self.assertNotIn("prompt",public["curriculum_selection"])
