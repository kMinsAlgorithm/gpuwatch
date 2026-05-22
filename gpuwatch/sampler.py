from __future__ import annotations

import time
from typing import Generator, Iterable, Optional

from gpuwatch.backends.fake import FakeGpuBackend
from gpuwatch.backends.host import HostBackend
from gpuwatch.backends.nvidia_smi import NvidiaSmiGpuBackend, NvidiaSmiUnavailable
from gpuwatch.backends.nvml import NvmlGpuBackend, NvmlUnavailable
from gpuwatch.models import SystemSnapshot


def sample_once(backend: str = "auto", fallback_to_fake: bool = False) -> SystemSnapshot:
    host_backend = HostBackend()
    host = host_backend.collect()
    errors = []
    backend_name = backend

    if backend == "fake":
        gpus = FakeGpuBackend().collect(host_backend)
        backend_name = "fake"
    elif backend == "smi":
        try:
            gpus = NvidiaSmiGpuBackend().collect(host_backend)
            backend_name = "nvidia-smi"
        except NvidiaSmiUnavailable as exc:
            errors.append(str(exc))
            gpus = ()
            backend_name = "nvidia-smi"
    else:
        try:
            with NvmlGpuBackend() as nvml_backend:
                gpus = nvml_backend.collect(host_backend)
            backend_name = "nvml"
        except NvmlUnavailable as exc:
            errors.append(str(exc))
            if backend == "nvml":
                gpus = ()
                backend_name = "nvml"
                return SystemSnapshot(host=host, gpus=tuple(gpus), errors=tuple(errors), backend=backend_name)
            try:
                gpus = NvidiaSmiGpuBackend().collect(host_backend)
                backend_name = "nvidia-smi"
            except NvidiaSmiUnavailable as smi_exc:
                errors.append(str(smi_exc))
                if fallback_to_fake:
                    gpus = FakeGpuBackend().collect(host_backend)
                    errors = []
                    backend_name = "fake"
                else:
                    gpus = ()
                    backend_name = "nvml"

    return SystemSnapshot(host=host, gpus=tuple(gpus), errors=tuple(errors), backend=backend_name)


def watch(
    interval: float = 1.0,
    backend: str = "auto",
    limit: Optional[int] = None,
    fallback_to_fake: bool = False,
) -> Generator[SystemSnapshot, None, None]:
    count = 0
    while True:
        started = time.monotonic()
        yield sample_once(backend=backend, fallback_to_fake=fallback_to_fake)
        count += 1
        if limit is not None and count >= limit:
            break
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, interval - elapsed))


def collect_processes(snapshot: SystemSnapshot) -> Iterable:
    return snapshot.all_processes()
