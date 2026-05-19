import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "gpuwatch.cli", *args],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    def test_once_json_filter_has_schema_and_training(self):
        result = self.run_cli("once", "--backend", "fake", "--gpu", "1", "--json")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema_version"], "0.2")
        self.assertEqual([gpu["index"] for gpu in payload["gpus"]], [1])
        self.assertIn("training", payload)

    def test_json_watch_includes_and_can_omit_training(self):
        result = self.run_cli("json", "--watch", "--limit", "1", "--backend", "fake")
        payload = json.loads(result.stdout.splitlines()[0])
        self.assertEqual(payload["schema_version"], "0.2")
        self.assertIn("training", payload)

        result = self.run_cli("json", "--watch", "--limit", "1", "--backend", "fake", "--no-training")
        payload = json.loads(result.stdout.splitlines()[0])
        self.assertNotIn("training", payload)

    def test_train_status_json_has_expanded_fields(self):
        result = self.run_cli("train-status", "--backend", "fake", "--json")
        payload = json.loads(result.stdout)
        if payload:
            self.assertIn("eta_seconds", payload[0])
            self.assertIn("state_reason", payload[0])

    def test_doctor_root_fake_reports_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "train.log").write_text("[INFO][demo][Epoch 1/2]\n", encoding="utf-8")
            result = self.run_cli("doctor", "--backend", "fake", "--root", tmp)
            self.assertIn("Project Roots", result.stdout)
            self.assertIn(tmp, result.stdout)


if __name__ == "__main__":
    unittest.main()
