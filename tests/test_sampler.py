import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gpuwatch.backends.host import _cpu_temperature
from gpuwatch.backends.nvidia_smi import NvidiaSmiUnavailable
from gpuwatch.backends.nvml import NvmlUnavailable
from gpuwatch.sampler import sample_once


def _sensor(label, current):
    return SimpleNamespace(label=label, current=current, high=None, critical=None)


class SamplerTests(unittest.TestCase):
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
