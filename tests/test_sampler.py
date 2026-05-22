import unittest
from unittest.mock import patch

from gpuwatch.backends.nvidia_smi import NvidiaSmiUnavailable
from gpuwatch.backends.nvml import NvmlUnavailable
from gpuwatch.sampler import sample_once


class SamplerTests(unittest.TestCase):
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
