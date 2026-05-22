import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from gpuwatch.models import GpuProcessSnapshot
from gpuwatch.training.status import _best_log_match, _parse_heartbeat, _has_project_marker, parse_log, snapshot
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

    def test_parse_5_20_start_and_epoch_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "520caer_prt_v1_20260521_001330_eth2_gpu1.log"
            log.write_text(
                "\n".join(
                    [
                        "[INFO][zara1] Start Training | epochs=150 | lr=0.0012",
                        "[ZARA1][TRAIN] Epoch 4",
                        "[INFO][zara1][seed 1][Epoch 4/149] loss(total/traj/distill/aux)=(0.30738/0.29932/0.00807/0.00000) | val_ADE=0.28166 | val_FDE=0.48439 | score=0.76605 | best_epoch=3",
                    ]
                )
            )
            status = parse_log(log)
            self.assertEqual(status.epoch, 4)
            self.assertEqual(status.max_epoch, 149)
            self.assertEqual(status.phase, "between_epochs")
            self.assertAlmostEqual(status.process_progress_percent, 100.0 * 5.0 / 149.0)
            self.assertEqual(status.learning_rate, 0.0012)
            self.assertEqual(status.loss, 0.30738)
            self.assertEqual(status.val_ade, 0.28166)
            self.assertEqual(status.val_fde, 0.48439)

    def test_parse_5_20_start_before_first_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "fresh.log"
            log.write_text("[INFO][zara1] Start Training | epochs=150 | lr=0.0012\n[ZARA1][TRAIN] Epoch 0\n")
            status = parse_log(log)
            self.assertEqual(status.epoch, 0)
            self.assertEqual(status.max_epoch, 149)
            self.assertEqual(status.phase, "train")
            self.assertEqual(status.epoch_label(), "0/149")
            self.assertEqual(status.learning_rate, 0.0012)

    def test_log_match_uses_run_name_tokens_for_5_20_nohup_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nba_log = root / "520caer_prt_v1_20260521_001330_nba_gpu0.log"
            eth2_log = root / "520caer_prt_v1_20260521_001330_eth2_gpu1.log"
            nba_log.write_text("[INFO][NBA][seed 0][Epoch 7/99] score=1.7\n")
            eth2_log.write_text("[INFO][zara1][seed 1][Epoch 4/149] score=0.7\n")
            statuses = [parse_log(nba_log), parse_log(eth2_log)]
            process = GpuProcessSnapshot(
                pid=2991475,
                gpu_index=1,
                gpu_uuid="GPU-test",
                cmdline=(
                    "python3",
                    "./eth_rg_hrt_v5_relation_logv1_agentwise_54edge.py",
                    "--run-name",
                    "520caer_prt_v1_eth2_seed1",
                    "--datasets",
                    "zara1,zara2",
                ),
            )
            match = _best_log_match(process, statuses)
            self.assertIsNotNone(match)
            self.assertIn("eth2_gpu1", match.log_path)

    def test_log_match_keeps_5_20_duplicate_run_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_log = root / "520caer_prt_v1_20260521_001330_eth2_gpu1.log"
            duplicate_log = root / "520caer_prt_v1_20260521_001543_eth2_duplicate_gpu1.log"
            original_log.write_text("[INFO][zara1][seed 1][Epoch 4/149] score=0.7\n")
            duplicate_log.write_text("[INFO][zara1][seed 1][Epoch 5/149] score=0.6\n")
            statuses = [parse_log(original_log), parse_log(duplicate_log)]
            original_process = GpuProcessSnapshot(
                pid=2991475,
                gpu_index=1,
                gpu_uuid="GPU-test",
                cmdline=("python3", "./eth.py", "--run-name", "520caer_prt_v1_eth2_seed1"),
            )
            duplicate_process = GpuProcessSnapshot(
                pid=3012627,
                gpu_index=1,
                gpu_uuid="GPU-test",
                cmdline=("python3", "./eth.py", "--run-name", "520caer_prt_v1_eth2_seed1_dup_20260521_001543"),
            )
            original_match = _best_log_match(original_process, statuses)
            duplicate_match = _best_log_match(duplicate_process, statuses)
            self.assertIsNotNone(original_match)
            self.assertIsNotNone(duplicate_match)
            self.assertIn("001330_eth2_gpu1", original_match.log_path)
            self.assertIn("001543_eth2_duplicate", duplicate_match.log_path)

    def test_log_match_respects_gpu_suffix_when_new_duplicate_log_is_fresher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eth2_log = root / "520caer_prt_v1_20260521_001543_eth2_duplicate_gpu1.log"
            hotel_log = root / "520caer_prt_v1_20260521_160323_hotel_duplicate_gpu0.log"
            eth2_log.write_text("[INFO][zara1][seed 1][Epoch 79/149] score=0.6\n")
            hotel_log.write_text("[INFO][hotel] Start Training | epochs=150 | lr=0.0018\n[HOTEL][TRAIN] Epoch 0\n")
            statuses = [parse_log(eth2_log), parse_log(hotel_log)]
            process = GpuProcessSnapshot(
                pid=3012627,
                gpu_index=1,
                gpu_uuid="GPU-test",
                cmdline=("python3", "./eth.py", "--run-name", "520caer_prt_v1_eth2_seed1_dup_20260521_001543"),
            )
            match = _best_log_match(process, statuses)
            self.assertIsNotNone(match)
            self.assertIn("eth2_duplicate_gpu1", match.log_path)

    def test_log_match_rejects_conflicting_zara_dataset_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zara1_log = root / "520caer_prt_v1_20260521_161530_zara1_resume_gpu1.log"
            zara2_log = root / "520caer_prt_v1_20260521_161037_zara2_gpu1.log"
            zara1_log.write_text("[INFO][zara1][seed 1][Epoch 80/149] score=0.6\n")
            zara2_log.write_text("[INFO][zara2][seed 1][Epoch 7/149] score=0.5\n")
            statuses = [parse_log(zara1_log), parse_log(zara2_log)]
            process = GpuProcessSnapshot(
                pid=3303779,
                gpu_index=1,
                gpu_uuid="GPU-test",
                cmdline=(
                    "python3",
                    "./eth.py",
                    "--run-name",
                    "520caer_prt_v1_zara2_seed1_gpu1_20260521_161037",
                    "--datasets",
                    "zara2",
                ),
            )
            match = _best_log_match(process, statuses)
            self.assertIsNotNone(match)
            self.assertIn("zara2_gpu1", match.log_path)

    def test_snapshot_infers_project_root_from_process_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            workdir = root / "scripts"
            workdir.mkdir()
            log_dir = root / "experiments" / "runs"
            log_dir.mkdir(parents=True)
            log = log_dir / "auto_run_gpu0.log"
            log.write_text("[INFO][auto_run][Epoch 3/9] score=0.7\n", encoding="utf-8")
            process = GpuProcessSnapshot(
                pid=2991475,
                gpu_index=0,
                gpu_uuid="GPU-test",
                cmdline=("python3", "train.py", "--run-name", "auto_run"),
                cwd=str(workdir),
            )

            statuses = snapshot(project_roots=[], processes=[process])

            self.assertEqual(len(statuses), 1)
            status = statuses[0]
            self.assertEqual(status.pid, process.pid)
            self.assertEqual(status.gpu_index, 0)
            self.assertEqual(status.run_name, "auto_run")
            self.assertEqual(status.epoch, 3)
            self.assertEqual(status.max_epoch, 9)
            self.assertIn("log-match", status.evidence)

    def test_project_marker_check_ignores_permission_errors(self):
        with patch("gpuwatch.training.status.Path.exists", side_effect=PermissionError("denied")):
            with patch("gpuwatch.training.status.Path.is_dir", side_effect=PermissionError("denied")):
                self.assertFalse(_has_project_marker(Path("/run/user/126/gdm")))

    def test_snapshot_does_not_bind_graphics_processes_to_training_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            log = root / "auto_run_gpu0.log"
            log.write_text("[INFO][auto_run][Epoch 3/9] score=0.7\n", encoding="utf-8")
            process = GpuProcessSnapshot(
                pid=5017,
                gpu_index=0,
                gpu_uuid="GPU-test",
                type="G",
                name="Xorg",
                cmdline=("/usr/lib/xorg/Xorg", "-auth", "/run/user/126/gdm/Xauthority"),
                cwd=str(root),
            )

            statuses = snapshot(project_roots=[], processes=[process])

            self.assertEqual(len(statuses), 1)
            status = statuses[0]
            self.assertEqual(status.pid, process.pid)
            self.assertEqual(status.state, "unbound")
            self.assertNotIn("log-match", status.evidence)

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

    def test_training_run_records_visible_gpu_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_run_dir = os.environ.get("GPUWATCH_RUN_DIR")
            previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["GPUWATCH_RUN_DIR"] = tmp
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            run = TrainingRun("gpu_hint", total_epochs=5, skill_dir=tmp)
            try:
                run.__enter__()
                run.step(epoch=1, step=2, total_steps=10)
                statuses = snapshot(project_roots=[])
            finally:
                run.__exit__(None, None, None)
                if previous_run_dir is None:
                    os.environ.pop("GPUWATCH_RUN_DIR", None)
                else:
                    os.environ["GPUWATCH_RUN_DIR"] = previous_run_dir
                if previous_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible
            self.assertTrue(statuses)
            self.assertEqual(statuses[0].gpu_index, 3)

    def test_training_run_records_ddp_rank_gpu_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = {key: os.environ.get(key) for key in ("GPUWATCH_RUN_DIR", "CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "RANK", "WORLD_SIZE")}
            os.environ["GPUWATCH_RUN_DIR"] = tmp
            os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
            os.environ["LOCAL_RANK"] = "1"
            os.environ["RANK"] = "5"
            os.environ["WORLD_SIZE"] = "8"
            run = TrainingRun("ddp_hint", total_epochs=5, skill_dir=tmp, project="demo")
            try:
                run.__enter__()
                run.step(epoch=1, step=2, total_steps=10)
                statuses = snapshot(project_roots=[])
            finally:
                run.__exit__(None, None, None)
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            self.assertEqual(statuses[0].gpu_index, 3)
            self.assertEqual(statuses[0].local_rank, 1)
            self.assertEqual(statuses[0].rank, 5)
            self.assertEqual(statuses[0].world_size, 8)
            self.assertEqual(statuses[0].project, "demo")

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

    def test_heartbeat_marks_missing_pid_orphaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_99999999.jsonl"
            path.write_text(
                '{"event": "step", "pid": 99999999, "run_name": "gone", "epoch": 4, "total_epochs": 9}\n',
                encoding="utf-8",
            )
            status = _parse_heartbeat(path)
            self.assertEqual(status.state, "orphaned")
            self.assertEqual(status.state_reason, "pid_missing")

    def test_heartbeat_preserves_metrics_and_computes_eta_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"run_{os.getpid()}.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"event": "run_start", "pid": %d, "run_name": "metrics", "total_epochs": 2, "project": "demo"}' % os.getpid(),
                        '{"event": "step", "pid": %d, "run_name": "metrics", "epoch": 0, "step": 10, "total_steps": 100, "time": 1000.0}' % os.getpid(),
                        '{"checkpoint_path": "/tmp/ckpt.pt", "epoch": 0, "event": "step", "learning_rate": 0.001, "loss": 0.42, "metric_name": "acc", "metric_value": 0.9, "pid": %d, "rank": 1, "run_name": "metrics", "step": 20, "time": 1005.0, "total_steps": 100, "world_size": 4}' % os.getpid(),
                    ]
                ),
                encoding="utf-8",
            )
            status = _parse_heartbeat(path)
            self.assertEqual(status.loss, 0.42)
            self.assertEqual(status.learning_rate, 0.001)
            self.assertEqual(status.checkpoint_path, "/tmp/ckpt.pt")
            self.assertEqual(status.metric_name, "acc")
            self.assertEqual(status.metric_value, 0.9)
            self.assertEqual(status.rank, 1)
            self.assertEqual(status.world_size, 4)
            self.assertEqual(status.speed_unit, "step")
            self.assertAlmostEqual(status.speed_per_second, 2.0)
            self.assertAlmostEqual(status.eta_seconds, 90.0)
            self.assertEqual(status.state, "stalled")
            self.assertEqual(status.state_reason, "heartbeat_stale")


if __name__ == "__main__":
    unittest.main()
