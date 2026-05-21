from __future__ import annotations

import csv
import subprocess
from io import StringIO
from typing import Dict, Iterable, List, Optional, Tuple

from gpuwatch.backends.host import HostBackend
from gpuwatch.models import GpuProcessSnapshot, GpuSnapshot


class NvidiaSmiUnavailable(RuntimeError):
    pass


GPU_QUERY = (
    "index,uuid,name,pci.bus_id,temperature.gpu,fan.speed,"
    "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit"
)
PROCESS_QUERY = "pid,gpu_uuid,used_memory"


def _run_nvidia_smi(args: List[str]) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NvidiaSmiUnavailable(f"nvidia-smi unavailable: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise NvidiaSmiUnavailable(f"nvidia-smi query failed: {detail}")
    return result.stdout


def _csv_rows(text: str) -> Iterable[List[str]]:
    for row in csv.reader(StringIO(text)):
        values = [item.strip() for item in row]
        if values and any(values):
            yield values


def _optional_int(value: str) -> Optional[int]:
    try:
        value = value.strip()
        if not value or value.upper() in {"N/A", "[N/A]", "[NOT SUPPORTED]"}:
            return None
        return int(float(value.split()[0]))
    except (TypeError, ValueError, IndexError):
        return None


def _optional_float(value: str) -> Optional[float]:
    try:
        value = value.strip()
        if not value or value.upper() in {"N/A", "[N/A]", "[NOT SUPPORTED]"}:
            return None
        return float(value.split()[0])
    except (TypeError, ValueError, IndexError):
        return None


class NvidiaSmiGpuBackend:
    name = "nvidia-smi"

    def collect(self, host_backend: HostBackend) -> Tuple[GpuSnapshot, ...]:
        gpu_rows = _run_nvidia_smi(
            [
                f"--query-gpu={GPU_QUERY}",
                "--format=csv,noheader,nounits",
            ]
        )
        process_rows = _collect_process_rows()
        processes_by_uuid = _processes_by_uuid(process_rows, host_backend)

        snapshots = []
        for values in _csv_rows(gpu_rows):
            if len(values) < 12:
                continue
            index = _optional_int(values[0])
            if index is None:
                continue
            uuid = values[1]
            process_infos = tuple(
                _attach_gpu_index(process, index, uuid)
                for process in processes_by_uuid.get(uuid, ())
            )
            snapshots.append(
                GpuSnapshot(
                    index=index,
                    uuid=uuid,
                    name=values[2] or "NVIDIA GPU",
                    bus_id=values[3] or None,
                    temperature_c=_optional_int(values[4]),
                    fan_percent=_optional_int(values[5]),
                    utilization_gpu_percent=_optional_int(values[6]),
                    utilization_memory_percent=_optional_int(values[7]),
                    memory_used_mb=_optional_int(values[8]),
                    memory_total_mb=_optional_int(values[9]),
                    power_draw_w=_optional_float(values[10]),
                    power_limit_w=_optional_float(values[11]),
                    processes=process_infos,
                )
            )
        return tuple(snapshots)


def _collect_process_rows() -> List[List[str]]:
    try:
        text = _run_nvidia_smi(
            [
                f"--query-compute-apps={PROCESS_QUERY}",
                "--format=csv,noheader,nounits",
            ]
        )
    except NvidiaSmiUnavailable:
        return []
    return list(_csv_rows(text))


def _processes_by_uuid(
    rows: Iterable[List[str]],
    host_backend: HostBackend,
) -> Dict[str, List[GpuProcessSnapshot]]:
    by_uuid: Dict[str, List[GpuProcessSnapshot]] = {}
    for values in rows:
        if len(values) < 3:
            continue
        pid = _optional_int(values[0])
        gpu_uuid = values[1]
        if pid is None or not gpu_uuid:
            continue
        snapshot = host_backend.enrich_process(
            pid=pid,
            gpu_index=-1,
            gpu_uuid=gpu_uuid,
            gpu_memory_mb=_optional_int(values[2]) or 0,
            process_type="C",
        )
        by_uuid.setdefault(gpu_uuid, []).append(snapshot)
    return by_uuid


def _attach_gpu_index(process: GpuProcessSnapshot, gpu_index: int, gpu_uuid: str) -> GpuProcessSnapshot:
    return GpuProcessSnapshot(
        pid=process.pid,
        gpu_index=gpu_index,
        gpu_uuid=gpu_uuid,
        gpu_memory_mb=process.gpu_memory_mb,
        type=process.type,
        name=process.name,
        username=process.username,
        cmdline=process.cmdline,
        cwd=process.cwd,
        cpu_percent=process.cpu_percent,
        rss_mb=process.rss_mb,
        create_time=process.create_time,
        elapsed_s=process.elapsed_s,
    )
