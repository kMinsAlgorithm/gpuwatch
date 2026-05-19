from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple


def _tuple_to_list(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_tuple_to_list(item) for item in value]
    if isinstance(value, dict):
        return {key: _tuple_to_list(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tuple_to_list(item) for item in value]
    return value


@dataclass(frozen=True)
class GpuProcessSnapshot:
    pid: int
    gpu_index: int
    gpu_uuid: str
    gpu_memory_mb: Optional[int] = None
    type: Optional[str] = None
    name: Optional[str] = None
    username: Optional[str] = None
    cmdline: Tuple[str, ...] = field(default_factory=tuple)
    cwd: Optional[str] = None
    cpu_percent: Optional[float] = None
    rss_mb: Optional[float] = None
    create_time: Optional[float] = None
    elapsed_s: Optional[float] = None

    @property
    def command(self) -> str:
        if self.cmdline:
            return " ".join(self.cmdline)
        return self.name or ""

    def to_dict(self) -> Dict[str, Any]:
        return _tuple_to_list(asdict(self))


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    uuid: str
    name: str
    bus_id: Optional[str] = None
    temperature_c: Optional[int] = None
    fan_percent: Optional[int] = None
    utilization_gpu_percent: Optional[int] = None
    utilization_memory_percent: Optional[int] = None
    memory_used_mb: Optional[int] = None
    memory_total_mb: Optional[int] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    processes: Tuple[GpuProcessSnapshot, ...] = field(default_factory=tuple)

    @property
    def memory_percent(self) -> Optional[float]:
        if not self.memory_total_mb:
            return None
        if self.memory_used_mb is None:
            return None
        return (self.memory_used_mb / self.memory_total_mb) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return _tuple_to_list(asdict(self))


@dataclass(frozen=True)
class HostSnapshot:
    timestamp: float
    hostname: str
    cpu_percent: Optional[float] = None
    load_avg: Tuple[float, float, float] = field(default_factory=tuple)
    memory_percent: Optional[float] = None
    memory_used_mb: Optional[int] = None
    memory_total_mb: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return _tuple_to_list(asdict(self))


@dataclass(frozen=True)
class SystemSnapshot:
    host: HostSnapshot
    gpus: Tuple[GpuSnapshot, ...] = field(default_factory=tuple)
    errors: Tuple[str, ...] = field(default_factory=tuple)
    backend: str = "auto"

    @property
    def timestamp(self) -> float:
        return self.host.timestamp

    def all_processes(self) -> Iterable[GpuProcessSnapshot]:
        for gpu in self.gpus:
            for process in gpu.processes:
                yield process

    def to_dict(self) -> Dict[str, Any]:
        return _tuple_to_list(asdict(self))
