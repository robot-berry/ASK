from __future__ import annotations

import unittest

from test8_ans.config import Settings


class SettingsTests(unittest.TestCase):
    def test_accepts_public_and_workspace_endpoints(self) -> None:
        Settings(
            api_key="key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).validate(require_api_key=True)
        Settings(
            api_key="key",
            base_url="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        ).validate(require_api_key=True)

    def test_rejects_endpoint_that_could_receive_the_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "官方域名"):
            Settings(
                api_key="key",
                base_url="https://example.com/compatible-mode/v1",
            ).validate(require_api_key=True)


if __name__ == "__main__":
    unittest.main()
