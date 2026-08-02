from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.disentangled_cvae_uwf_zeekdata24.download import download_dataset
from experiments.disentangled_cvae_uwf_zeekdata24.tests.helpers import csv_bytes, uwf_row


class DownloadTests(unittest.TestCase):
    def test_download_manifest_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            categories = [
                "Benign", "Credential_Access", "Defense_Evasion", "Exfiltration",
                "Initial_Access", "Persistence", "Privilege_Escalation", "Reconnaissance",
            ]
            payload = csv_bytes([uwf_row("u1")])

            def fetch(url: str) -> bytes:
                if url.endswith("/"):
                    return b'<a href="part-test.csv">part-test.csv</a>'
                return payload

            manifest_path = download_dataset(
                {"raw_dir": str(root), "source_base_url": "https://example.test", "source_categories": categories},
                fetch_bytes=fetch,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["files"]), 8)
            self.assertEqual(manifest["license"], "CC BY 4.0")
            self.assertTrue(all((root / f"{category}.csv").is_file() for category in categories))

    def test_download_rejects_invalid_official_schema(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            categories = [
                "Benign", "Credential_Access", "Defense_Evasion", "Exfiltration",
                "Initial_Access", "Persistence", "Privilege_Escalation", "Reconnaissance",
            ]

            def fetch(url: str) -> bytes:
                return b'<a href="part-test.csv">part-test.csv</a>' if url.endswith("/") else b"uid,proto\nu1,tcp\n"

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                download_dataset(
                    {
                        "raw_dir": folder,
                        "source_base_url": "https://example.test",
                        "source_categories": categories,
                    },
                    fetch_bytes=fetch,
                )


if __name__ == "__main__":
    unittest.main()
