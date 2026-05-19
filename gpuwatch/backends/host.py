from __future__ import annotations

import os
import socket
import time
from typing import Optional, Sequence, Tuple

import psutil

from gpuwatch.models import GpuProcessSnapshot, HostSnapshot


def _mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value) / (1024.0 * 1024.0), 1)


class HostBackend:
    """Collect host and process metadata through psutil."""

    def collect(self) -> HostSnapshot:
        memory = psutil.virtual_memory()
        try:
            load_avg = os.getloadavg()
        except (AttributeError, OSError):
            load_avg = ()
        return HostSnapshot(
            timestamp=time.time(),
            hostname=socket.gethostname(),
            cpu_percent=psutil.cpu_percent(interval=None),
            load_avg=tuple(float(item) for item in load_avg),
            memory_percent=float(memory.percent),
            memory_used_mb=int(memory.used // (1024 * 1024)),
            memory_total_mb=int(memory.total // (1024 * 1024)),
        )

    def enrich_process(
        self,
        pid: int,
        gpu_index: int,
        gpu_uuid: str,
        gpu_memory_mb: Optional[int],
        process_type: Optional[str],
    ) -> GpuProcessSnapshot:
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                cmdline = tuple(_safe_cmdline(process))
                create_time = process.create_time()
                elapsed_s = max(0.0, time.time() - create_time)
                memory_info = process.memory_info()
                return GpuProcessSnapshot(
                    pid=pid,
                    gpu_index=gpu_index,
                    gpu_uuid=gpu_uuid,
                    gpu_memory_mb=gpu_memory_mb,
                    type=process_type,
                    name=_safe_call(process.name),
                    username=_safe_call(process.username),
                    cmdline=cmdline,
                    cwd=_safe_call(process.cwd),
                    cpu_percent=_safe_call(process.cpu_percent),
                    rss_mb=_mb(memory_info.rss),
                    create_time=create_time,
                    elapsed_s=elapsed_s,
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return GpuProcessSnapshot(
                pid=pid,
                gpu_index=gpu_index,
                gpu_uuid=gpu_uuid,
                gpu_memory_mb=gpu_memory_mb,
                type=process_type,
            )


def _safe_call(func):
    try:
        return func()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        return None


def _safe_cmdline(process: psutil.Process) -> Sequence[str]:
    try:
        return process.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        name = _safe_call(process.name)
        return [name] if name else []
