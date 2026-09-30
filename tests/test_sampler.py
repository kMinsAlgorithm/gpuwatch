import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from gpuwatch.backends.host import _cpu_temperature
from gpuwatch.backends.nvidia_smi import NvidiaSmiUnavailable
from gpuwatch.backends.nvml import NvmlUnavailable
from gpuwatch.sampler import sample_once, with_training_storage
from gpuwatch.training.status import TrainingStatus


def _sensor(label, current):
    return SimpleNamespace(label=label, current=current, high=None, critical=None)


class SamplerTests(unittest.TestCase):
    def test_storage_prefers_checkpoint_volume_over_log_volume(self):
        snapshot = sample_once(backend="fake")
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "logs").mkdir()
            (project / "checkpoints").mkdir()

            def disk_usage(path):
                free = 5 if "checkpoints" in path else 50
                return SimpleNamespace(total=100, used=100 - free, free=free)

            status = TrainingStatus(
                project=str(project),
                checkpoint_path="checkpoints/model.pt",
                log_path=str(project / "logs" / "train.log"),
            )
            with patch("gpuwatch.backends.host.psutil.disk_usage", side_effect=disk_usage):
                selected = with_training_storage(snapshot, [status], [str(project)])
            self.assertEqual(selected.host.storage.free_bytes, 5)
            self.assertEqual(selected.host.storage.used_percent, 95.0)

    def test_missing_checkpoint_volume_falls_back_to_project_root(self):
        snapshot = sample_once(backend="fake")
        with TemporaryDirectory() as tmp:
            status = TrainingStatus(checkpoint_path="/__gpuwatch_missing_volume__/model.pt")
            usage = SimpleNamespace(total=100, used=70, free=30)
            with patch("gpuwatch.backends.host.psutil.disk_usage", return_value=usage) as read_usage:
                selected = with_training_storage(snapshot, [status], [tmp])
            read_usage.assert_called_once_with(tmp)
            self.assertEqual(selected.host.storage.free_bytes, 30)

    def test_cpu_temperature_prefers_intel_package_sensor(self):
        sensors = {
            "nvme": [_sensor("Composite", 48.9)],
            "coretemp": [_sensor("Package id 0", 69.0), _sensor("Core 0", 71.0)],
        }
        with patch("gpuwatch.backends.host.psutil.sensors_temperatures", return_value=sensors, create=True):
            self.assertEqual(_cpu_temperature(), 69.0)

    def test_cpu_temperature_reads_amd_tctl(self):
        sensors = {"k10temp": [_sensor("Tccd1", 60.0), _sensor("Tctl", 74.25)]}
        with patch("gpuwatch.backends.host.psutil.sensors_temperatures", return_value=sensors, create=True):
            self.assertEqual(_cpu_temperature(), 74.2)

    def test_cpu_temperature_is_none_without_cpu_sensor(self):
        with patch("gpuwatch.backends.host.psutil.sensors_temperatures", return_value={"nvme": [_sensor("", 40.0)]}, create=True):
            self.assertIsNone(_cpu_temperature())
        with patch("gpuwatch.backends.host.psutil.sensors_temperatures", side_effect=AttributeError, create=True):
            self.assertIsNone(_cpu_temperature())

    def test_auto_can_fall_back_to_fake_when_real_backends_fail(self):
        with patch("gpuwatch.sampler.NvmlGpuBackend", side_effect=NvmlUnavailable("nvml down")):
            with patch("gpuwatch.sampler.NvidiaSmiGpuBackend") as smi_backend:
                smi_backend.return_value.collect.side_effect = NvidiaSmiUnavailable("smi down")
                snapshot = sample_once(backend="auto", fallback_to_fake=True)

        self.assertEqual(snapshot.backend, "fake")
        self.assertEqual(len(snapshot.gpus), 4)
        self.assertFalse(snapshot.errors)

    def test_auto_preserves_backend_errors_when_fake_fallback_is_disabled(self):
        with patch("gpuwatch.sampler.NvmlGpuBackend", side_effect=NvmlUnavailable("nvml down")):
            with patch("gpuwatch.sampler.NvidiaSmiGpuBackend") as smi_backend:
                smi_backend.return_value.collect.side_effect = NvidiaSmiUnavailable("smi down")
                snapshot = sample_once(backend="auto", fallback_to_fake=False)

        self.assertEqual(snapshot.backend, "nvml")
        self.assertEqual(snapshot.gpus, ())
        self.assertEqual(snapshot.errors, ("nvml down", "smi down"))


if __name__ == "__main__":
    unittest.main()
