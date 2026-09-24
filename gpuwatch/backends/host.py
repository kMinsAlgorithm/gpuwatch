from __future__ import annotations

import os
import socket
import time
from typing import Dict, Optional, Sequence, Tuple

import psutil

from gpuwatch.models import GpuProcessSnapshot, HostSnapshot

# psutil's cpu_percent() measures the delta since the previous call on the same Process
# object, so keep objects alive across samples or every reading is 0.0.
_PROCESS_CACHE: Dict[int, psutil.Process] = {}
_PROCESS_CACHE_MAX = 4096

# Package-level CPU sensors by psutil driver name, most specific first: Intel coretemp,
# AMD k10temp/zenpower, and common ARM SoC drivers.
_CPU_SENSOR_LABELS = (
    ("coretemp", ("package id",)),
    ("k10temp", ("tctl", "tdie")),
    ("zenpower", ("tctl", "tdie")),
    ("cpu_thermal", ("",)),
    ("soc_thermal", ("",)),
)


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
            cpu_temperature_c=_cpu_temperature(),
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
            process = _cached_process(pid)
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
            _PROCESS_CACHE.pop(pid, None)
            return GpuProcessSnapshot(
                pid=pid,
                gpu_index=gpu_index,
                gpu_uuid=gpu_uuid,
                gpu_memory_mb=gpu_memory_mb,
                type=process_type,
            )


def _cpu_temperature() -> Optional[float]:
    """Hottest CPU package temperature in Celsius, or None when no CPU sensor is exposed."""
    try:
        sensors = psutil.sensors_temperatures()
    except (AttributeError, OSError, RuntimeError):
        return None
    for driver, label_prefixes in _CPU_SENSOR_LABELS:
        entries = sensors.get(driver) or ()
        readings = [
            entry.current
            for entry in entries
            if entry.current is not None
            and any((entry.label or "").lower().startswith(prefix) for prefix in label_prefixes)
        ]
        if not readings and driver == "coretemp":
            # Some kernels expose only per-core readings without a package label.
            readings = [entry.current for entry in entries if entry.current is not None]
        if readings:
            return round(float(max(readings)), 1)
    return None


def _cached_process(pid: int) -> psutil.Process:
    process = _PROCESS_CACHE.get(pid)
    # is_running() also compares create_time, so a recycled PID gets a fresh object.
    if process is not None and process.is_running():
        return process
    if len(_PROCESS_CACHE) >= _PROCESS_CACHE_MAX:
        _PROCESS_CACHE.clear()
    process = psutil.Process(pid)
    _PROCESS_CACHE[pid] = process
    return process


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
