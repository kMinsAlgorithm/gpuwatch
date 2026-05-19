import unittest

from gpuwatch.sampler import sample_once


class FakeBackendTests(unittest.TestCase):
    def test_fake_backend_has_gpus_and_processes(self):
        snapshot = sample_once(backend="fake")
        self.assertEqual(snapshot.backend, "fake")
        self.assertEqual(len(snapshot.gpus), 4)
        self.assertTrue(any(gpu.processes for gpu in snapshot.gpus))
        self.assertFalse(snapshot.errors)


if __name__ == "__main__":
    unittest.main()
