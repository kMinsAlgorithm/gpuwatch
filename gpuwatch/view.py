from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence, Tuple

from gpuwatch.models import GpuProcessSnapshot, GpuSnapshot, SystemSnapshot
from gpuwatch.training.status import TrainingStatus


@dataclass(frozen=True)
class ViewOptions:
    gpu_indices: Optional[Tuple[int, ...]] = None
    pids: Optional[Tuple[int, ...]] = None
    user: Optional[str] = None
    cmd: Optional[str] = None
    states: Optional[Tuple[str, ...]] = None
    run: Optional[str] = None
    dataset: Optional[str] = None
    failed_only: bool = False
    sort: str = "index"
    reverse: bool = False


def parse_int_list(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    if value is None or value == "":
        return None
    items = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_raw, end_raw = chunk.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            step = 1 if end >= start else -1
            items.extend(range(start, end + step, step))
        else:
            items.append(int(chunk))
    return tuple(dict.fromkeys(items))


def parse_str_list(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if value is None or value == "":
        return None
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def apply_view(
    snapshot: SystemSnapshot,
    statuses: Sequence[TrainingStatus],
    options: ViewOptions,
) -> tuple[SystemSnapshot, list[TrainingStatus]]:
    status_list = _filter_statuses(list(statuses), options)
    gpus = _filter_gpus(tuple(snapshot.gpus), options)
    gpus = _sort_gpus(gpus, options)
    status_list = _sort_statuses(status_list, options)
    return replace(snapshot, gpus=gpus), status_list


def _filter_gpus(gpus: Tuple[GpuSnapshot, ...], options: ViewOptions) -> Tuple[GpuSnapshot, ...]:
    filtered = []
    process_filter_active = any((options.pids, options.user, options.cmd))
    for gpu in gpus:
        if options.gpu_indices is not None and gpu.index not in options.gpu_indices:
            continue
        processes = tuple(process for process in gpu.processes if _process_matches(process, options))
        if process_filter_active and not processes:
            continue
        filtered.append(replace(gpu, processes=processes))
    return tuple(filtered)


def _filter_statuses(statuses: list[TrainingStatus], options: ViewOptions) -> list[TrainingStatus]:
    out = []
    for status in statuses:
        if options.gpu_indices is not None and status.gpu_index not in options.gpu_indices:
            continue
        if options.pids is not None and status.pid not in options.pids:
            continue
        if options.states is not None and (status.state or "").lower() not in options.states:
            continue
        if options.failed_only and (status.state or "").lower() not in ("failed", "stalled", "orphaned"):
            continue
        if options.run and options.run.lower() not in (status.run_name or "").lower():
            continue
        if options.dataset and options.dataset.lower() not in (status.dataset or "").lower():
            continue
        out.append(status)
    return out


def _process_matches(process: GpuProcessSnapshot, options: ViewOptions) -> bool:
    if options.pids is not None and process.pid not in options.pids:
        return False
    if options.user and options.user.lower() not in (process.username or "").lower():
        return False
    if options.cmd and options.cmd.lower() not in process.command.lower():
        return False
    return True


def _sort_gpus(gpus: Tuple[GpuSnapshot, ...], options: ViewOptions) -> Tuple[GpuSnapshot, ...]:
    sort = options.sort
    if sort == "util":
        key = lambda gpu: _none_last(gpu.utilization_gpu_percent)
    elif sort in ("vram", "mem"):
        key = lambda gpu: _none_last(gpu.memory_used_mb)
    elif sort == "temp":
        key = lambda gpu: _none_last(gpu.temperature_c)
    else:
        key = lambda gpu: (gpu.index,)
    return tuple(sorted(gpus, key=lambda gpu: (key(gpu), gpu.index), reverse=options.reverse))


def _sort_statuses(statuses: list[TrainingStatus], options: ViewOptions) -> list[TrainingStatus]:
    sort = options.sort
    if sort == "age":
        key = lambda status: _none_last(status.age_seconds if status.age_seconds is not None else status.stale_seconds)
    elif sort == "progress":
        key = lambda status: _none_last(status.process_progress_percent)
    elif sort == "eta":
        key = lambda status: _none_last(status.eta_seconds)
    elif sort == "run":
        key = lambda status: ((status.run_name or "").lower(),)
    elif sort == "dataset":
        key = lambda status: ((status.dataset or "").lower(),)
    else:
        key = lambda status: _none_last(status.gpu_index)
    return sorted(statuses, key=lambda status: (key(status), status.pid or -1), reverse=options.reverse)


def _none_last(value) -> tuple[int, object]:
    return (1, "") if value is None else (0, value)
