from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Generator, Iterable, Optional

from gpuwatch.backends.fake import FakeGpuBackend
from gpuwatch.backends.host import HostBackend, collect_storage
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


def with_training_storage(snapshot: SystemSnapshot, statuses: Iterable, project_roots: Iterable[str] = ()) -> SystemSnapshot:
    """Prefer the volume receiving checkpoints or logs over the launch directory."""
    statuses = sorted(statuses, key=lambda status: (getattr(status, "state", "") or "").lower() in {"complete", "orphaned"})
    roots = list(project_roots)
    process_dirs = [process.cwd for process in snapshot.all_processes() if process.cwd]
    fallback_base = roots[0] if roots else process_dirs[0] if process_dirs else None
    candidates = []
    for status in statuses:
        checkpoint = getattr(status, "checkpoint_path", None)
        if checkpoint:
            checkpoint_path = Path(checkpoint).expanduser()
            project = getattr(status, "project", None)
            if not checkpoint_path.is_absolute():
                base = project if project and Path(project).is_absolute() else fallback_base
                if base:
                    checkpoint_path = Path(base) / checkpoint_path
            candidates.append(str(checkpoint_path))
    for status in statuses:
        log_path = getattr(status, "log_path", None) or getattr(status, "events_path", None)
        if log_path:
            candidates.append(log_path)
    candidates.extend(roots)
    candidates.extend(process_dirs)
    for path in candidates:
        storage = collect_storage(path)
        if storage is not None:
            return replace(snapshot, host=replace(snapshot.host, storage=storage))
    return snapshot
