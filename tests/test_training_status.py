import tempfile
import unittest
import os
from pathlib import Path

from gpuwatch.training.status import _parse_heartbeat, parse_log, snapshot
from gpuwatch.training.tracker import TrainingRun


class TrainingStatusTests(unittest.TestCase):
    def test_parse_info_epoch_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text(
                "\n".join(
                    [
                        "[ETH][TRAIN] Epoch 2",
                        "[INFO][eth_seed1][Epoch 2/100] train_loss: 0.12 | val_FDE: 1.23 | val_ADE: 0.45 | score=1.68 | best_epoch: 2",
                    ]
                )
            )
            status = parse_log(log)
            self.assertEqual(status.run_name, "eth_seed1")
            self.assertEqual(status.epoch, 2)
            self.assertEqual(status.max_epoch, 100)
            self.assertEqual(status.val_ade, 0.45)
            self.assertEqual(status.val_fde, 1.23)
            self.assertEqual(status.best_epoch, 2)

    def test_parse_iteration_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "iter.log"
            log.write_text("[ETH][TRAIN] Epochs: 02/99| It: 0040/0800 | Loss: 0.1\n")
            status = parse_log(log)
            self.assertEqual(status.epoch, 2)
            self.assertEqual(status.max_epoch, 99)
            self.assertEqual(status.step, 40)
            self.assertEqual(status.total_steps, 800)
            self.assertEqual(status.phase, "train")

    def test_training_run_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("GPUWATCH_EXTRA_RUN_DIRS")
            previous_run_dir = os.environ.get("GPUWATCH_RUN_DIR")
            os.environ["GPUWATCH_EXTRA_RUN_DIRS"] = tmp
            os.environ["GPUWATCH_RUN_DIR"] = tmp
            run = TrainingRun("unit_run", total_epochs=10, skill_dir=tmp)
            try:
                run.__enter__()
                run.step(epoch=3, step=5, total_steps=10, val_ade=0.42)
                statuses = snapshot(project_roots=[])
            finally:
                run.__exit__(None, None, None)
                if previous is None:
                    os.environ.pop("GPUWATCH_EXTRA_RUN_DIRS", None)
                else:
                    os.environ["GPUWATCH_EXTRA_RUN_DIRS"] = previous
                if previous_run_dir is None:
                    os.environ.pop("GPUWATCH_RUN_DIR", None)
                else:
                    os.environ["GPUWATCH_RUN_DIR"] = previous_run_dir
            self.assertTrue(statuses)
            status = statuses[0]
            self.assertEqual(status.run_name, "unit_run")
            self.assertEqual(status.epoch, 3)
            self.assertEqual(status.step, 5)
            self.assertEqual(status.total_steps, 10)
            self.assertEqual(status.phase, "train")

    def test_heartbeat_ignores_partial_json_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_123.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"event": "run_start", "pid": 123, "run_name": "partial", "total_epochs": 10}',
                        '{"event": "step", "pid": 123, "run_name": "partial", "epoch": 4, "step": 2, "total_steps": 5}',
                        '{"event": "step", "pid":',
                    ]
                ),
                encoding="utf-8",
            )
            status = _parse_heartbeat(path)
            self.assertEqual(status.pid, 123)
            self.assertEqual(status.run_name, "partial")
            self.assertEqual(status.epoch, 4)
            self.assertEqual(status.step, 2)


if __name__ == "__main__":
    unittest.main()
