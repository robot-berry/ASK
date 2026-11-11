"""Prompt persistence and real portal authorization, without paid model calls."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_portal import portal, TOKEN
from fastapi.testclient import TestClient
import prompt_store


class PromptSettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict('os.environ', {'TEST9_PROMPT_DB': str(Path(self.tmp.name) / 'prompts.sqlite3')})
        self.env.start()
        self.client = TestClient(portal.app)
        self.headers = {'Authorization': 'Bearer '+TOKEN, 'X-Prompt-Editor': '1'}

    def tearDown(self):
        self.client.close()
        self.env.stop()
        self.tmp.cleanup()

    def test_save_restore_conflict_and_subject_isolation(self):
        url = '/api/v1/prompt-settings/math'
        before = self.client.get(url, headers=self.headers).json()
        physics = prompt_store.current('physics')
        text = before['text']+'\n测试说明：保留准确的中间关系。'
        result = self.client.put(url, headers=self.headers, json={'text':text, 'revision':before['revision']})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(prompt_store.current('math')['text'], text)
        self.assertEqual(prompt_store.current('physics'), physics)
        self.assertNotEqual(before['text'], text)
        self.assertEqual(self.client.put(url, headers=self.headers, json={'text':text+'修改', 'revision':before['revision']}).status_code, 409)
        history = self.client.get(url, headers=self.headers).json()['history']
        self.assertEqual(len(history), 2)
        restored = self.client.put(url, headers=self.headers, json={'text':history[-1]['text'], 'revision':result.json()['revision']})
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(prompt_store.current('math')['text'], before['text'])

    def test_owner_only_and_invalid_body(self):
        url = '/api/v1/prompt-settings/math'
        self.assertEqual(self.client.get(url).status_code, 403)
        device = 'test-device-not-owner'
        with patch.object(portal, 'DEVICE_TOKEN_HASHES', (hashlib.sha256(device.encode()).hexdigest(),)):
            self.assertEqual(self.client.get(url, headers={'Authorization':'Bearer '+device}).status_code, 403)
        self.assertEqual(self.client.put(url, headers={'Authorization':'Bearer '+TOKEN}, json={}).status_code, 415)
        self.assertEqual(self.client.put(url, headers=self.headers, json={'text':'too short'}).status_code, 400)
        self.assertEqual(self.client.get('/api/v1/prompt-settings/unknown', headers=self.headers).status_code, 400)

    def test_loopback_no_token_mode_can_read_and_save(self):
        with patch.object(portal, 'ALLOW_INSECURE_NO_AUTH', True), patch.object(portal, 'ACCESS_TOKEN', ''):
            with TestClient(portal.app, base_url='http://127.0.0.1', client=('127.0.0.1', 12345)) as client:
                url = '/api/v1/prompt-settings/math'
                before = client.get(url)
                self.assertEqual(before.status_code, 200)
                data = before.json()
                result = client.put(url, headers={'X-Prompt-Editor':'1'}, json={'text':data['text'], 'revision':data['revision']})
                self.assertEqual(result.status_code, 200)
            with TestClient(portal.app, base_url='http://10.10.16.247', client=('10.10.16.200', 12345)) as client:
                self.assertEqual(client.get('/api/v1/prompt-settings/math').status_code, 403)

    def test_browser_login_and_full_markdown_import(self):
        self.assertEqual(self.client.post('/api/v1/auth/login', json={'token':TOKEN}).status_code, 200)
        url='/api/v1/prompt-settings/math'
        before=self.client.get(url).json()
        raw=(prompt_store.ROOT/'math/source'/prompt_store.FILES['math']).read_text(encoding='utf-8')
        result=self.client.put(url, headers={'X-Prompt-Editor':'1'}, json={'text':raw, 'revision':before['revision']})
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.json()['text'],before['text'])
        self.assertEqual(self.client.get('/prompt-settings').status_code,200)


if __name__=='__main__':
    unittest.main()
