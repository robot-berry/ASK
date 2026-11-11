from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from provision_device_tokens import generate_records, write_outputs


class ProvisionDeviceTokensTests(unittest.TestCase):
    def test_generates_unique_device_records_and_matching_hashes(self) -> None:
        generated = iter(("alpha", "bravo", "charlie"))
        with patch(
            "provision_device_tokens.secrets.token_urlsafe",
            side_effect=lambda _size: next(generated),
        ):
            records = generate_records(3, "ASK-UNIT")

        self.assertEqual(
            [record["device_sn"] for record in records],
            ["ASK-UNIT-0001", "ASK-UNIT-0002", "ASK-UNIT-0003"],
        )
        self.assertEqual(records[0]["bearer_token"], "t6_alpha")
        self.assertEqual(
            records[0]["token_sha256"],
            hashlib.sha256(b"t6_alpha").hexdigest(),
        )

    def test_writes_private_inventory_and_server_only_hash_setting(self) -> None:
        records = generate_records(2, "DEVICE")
        with tempfile.TemporaryDirectory() as temporary:
            credentials, hashes = write_outputs(
                Path(temporary), records, force=False
            )
            with credentials.open(encoding="utf-8-sig", newline="") as handle:
                persisted = list(csv.DictReader(handle))
            setting = hashes.read_text(encoding="utf-8")

            self.assertEqual(persisted, records)
            self.assertTrue(setting.startswith("TEST6_DEVICE_TOKEN_HASHES="))
            self.assertNotIn(records[0]["bearer_token"], setting)
            with self.assertRaises(FileExistsError):
                write_outputs(Path(temporary), records, force=False)


if __name__ == "__main__":
    unittest.main()
