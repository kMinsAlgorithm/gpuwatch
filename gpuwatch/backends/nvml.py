from __future__ import annotations

from typing import Dict, Optional, Tuple

from gpuwatch.backends.host import HostBackend
from gpuwatch.models import GpuProcessSnapshot, GpuSnapshot


class NvmlUnavailable(RuntimeError):
    pass


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _bytes_to_mb(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        raw = int(value)
        if raw < 0 or raw > (1 << 60):
            return None
        return int(round(raw / (1024.0 * 1024.0)))
    except (TypeError, ValueError):
        return None


class NvmlGpuBackend:
    name = "nvml"

    def __init__(self) -> None:
        try:
            import pynvml  # type: ignore
        except Exception as exc:
            raise NvmlUnavailable(
                "NVML Python bindings are unavailable. Install nvidia-ml-py."
            ) from exc
        self.nvml = pynvml
        self.initialized = False

    def __enter__(self) -> "NvmlGpuBackend":
        if not self.initialized:
            try:
                self.nvml.nvmlInit()
            except Exception as exc:
                raise NvmlUnavailable(f"NVML init failed: {exc}") from exc
            self.initialized = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self.initialized:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
            self.initialized = False

    def collect(self, host_backend: HostBackend) -> Tuple[GpuSnapshot, ...]:
        if not self.initialized:
            self.__enter__()
        try:
            count = int(self.nvml.nvmlDeviceGetCount())
        except Exception as exc:
            raise NvmlUnavailable(f"NVML device query failed: {exc}") from exc
        snapshots = []
        for index in range(count):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(index)
            uuid = _decode(self.nvml.nvmlDeviceGetUUID(handle))
            snapshots.append(self._collect_device(index, handle, uuid, host_backend))
        return tuple(snapshots)

    def _collect_device(
        self,
        index: int,
        handle,
        uuid: str,
        host_backend: HostBackend,
    ) -> GpuSnapshot:
        memory = self._safe_call(self.nvml.nvmlDeviceGetMemoryInfo, handle)
        utilization = self._safe_call(self.nvml.nvmlDeviceGetUtilizationRates, handle)
        pci = self._safe_call(self.nvml.nvmlDeviceGetPciInfo, handle)
        power_draw = self._safe_call(self.nvml.nvmlDeviceGetPowerUsage, handle)
        power_limit = self._safe_call(self.nvml.nvmlDeviceGetPowerManagementLimit, handle)

        processes = self._collect_processes(index, handle, uuid, host_backend)
        return GpuSnapshot(
            index=index,
            uuid=uuid,
            name=_decode(self._safe_call(self.nvml.nvmlDeviceGetName, handle) or "NVIDIA GPU"),
            bus_id=_decode(getattr(pci, "busId", "")) if pci is not None else None,
            temperature_c=self._safe_call(
                self.nvml.nvmlDeviceGetTemperature,
                handle,
                self.nvml.NVML_TEMPERATURE_GPU,
            ),
            fan_percent=self._safe_call(self.nvml.nvmlDeviceGetFanSpeed, handle),
            utilization_gpu_percent=getattr(utilization, "gpu", None),
            utilization_memory_percent=getattr(utilization, "memory", None),
            memory_used_mb=_bytes_to_mb(getattr(memory, "used", None)),
            memory_total_mb=_bytes_to_mb(getattr(memory, "total", None)),
            power_draw_w=round(float(power_draw) / 1000.0, 1) if power_draw is not None else None,
            power_limit_w=round(float(power_limit) / 1000.0, 1) if power_limit is not None else None,
            processes=processes,
        )

    def _collect_processes(
        self,
        index: int,
        handle,
        uuid: str,
        host_backend: HostBackend,
    ) -> Tuple[GpuProcessSnapshot, ...]:
        by_pid: Dict[int, Dict[str, object]] = {}
        for process_type, function_names in (
            ("C", ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses")),
            ("G", ("nvmlDeviceGetGraphicsRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses")),
        ):
            infos = None
            for function_name in function_names:
                function = getattr(self.nvml, function_name, None)
                if function is None:
                    continue
                infos = self._safe_call(function, handle)
                if infos is not None:
                    break
            if not infos:
                continue
            for info in infos:
                pid = int(getattr(info, "pid"))
                used_memory = _bytes_to_mb(getattr(info, "usedGpuMemory", None))
                record = by_pid.setdefault(pid, {"memory": 0, "type": set()})
                if used_memory is not None:
                    record["memory"] = int(record["memory"]) + used_memory
                record_type = record["type"]
                if hasattr(record_type, "add"):
                    record_type.add(process_type)

        snapshots = []
        for pid, record in sorted(by_pid.items()):
            process_types = record.get("type")
            type_label = "+".join(sorted(process_types)) if process_types else None
            snapshots.append(
                host_backend.enrich_process(
                    pid=pid,
                    gpu_index=index,
                    gpu_uuid=uuid,
                    gpu_memory_mb=int(record.get("memory") or 0),
                    process_type=type_label,
                )
            )
        return tuple(snapshots)

    def _safe_call(self, function, *args):
        try:
            return function(*args)
        except Exception:
            return None
