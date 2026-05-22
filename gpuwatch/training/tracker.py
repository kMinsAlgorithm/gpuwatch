from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def default_run_dir() -> Path:
    return Path(os.environ.get("GPUWATCH_RUN_DIR", Path.home() / ".cache" / "gpuwatch" / "runs"))


def _parse_gpu_index(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    first = value.split(",", 1)[0].strip()
    if not first or first == "-1":
        return None
    try:
        return int(first)
    except ValueError:
        return None


def _parse_gpu_indices(value: Optional[str]) -> List[int]:
    if not value:
        return []
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item or item == "-1":
            continue
        try:
            indices.append(int(item))
        except ValueError:
            continue
    return indices


def _env_int(name: str) -> Optional[int]:
    try:
        value = os.environ.get(name)
        return int(value) if value is not None and value != "" else None
    except ValueError:
        return None


def _default_gpu_indices() -> List[int]:
    explicit = _parse_gpu_indices(os.environ.get("GPUWATCH_GPU_INDEX"))
    if explicit:
        return explicit
    visible = _parse_gpu_indices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    local_rank = _env_int("LOCAL_RANK")
    if visible and local_rank is not None and 0 <= local_rank < len(visible):
        return [visible[local_rank]]
    return visible[:1]


def _default_gpu_index() -> Optional[int]:
    indices = _default_gpu_indices()
    return indices[0] if indices else None


class TrainingRun:
    """Small heartbeat writer for training loops."""

    def __init__(
        self,
        run_name: str,
        total_epochs: Optional[int] = None,
        gpu_index: Optional[int] = None,
        gpu_indices: Optional[Iterable[int]] = None,
        dataset: Optional[str] = None,
        project: Optional[str] = None,
        skill_dir: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_name = run_name
        self.total_epochs = total_epochs
        inferred_indices = [int(item) for item in gpu_indices] if gpu_indices is not None else _default_gpu_indices()
        if gpu_index is not None and gpu_index not in inferred_indices:
            inferred_indices.insert(0, int(gpu_index))
        self.gpu_indices = tuple(inferred_indices)
        self.gpu_index = int(gpu_index) if gpu_index is not None else (self.gpu_indices[0] if self.gpu_indices else _default_gpu_index())
        self.dataset = dataset
        self.project = project
        self.metadata = metadata or {}
        base_dir = Path(skill_dir) if skill_dir else default_run_dir()
        self.run_dir = base_dir
        self.path = self.run_dir / f"run_{os.getpid()}.jsonl"
        self.pid = os.getpid()
        self.rank = _env_int("RANK")
        self.local_rank = _env_int("LOCAL_RANK")
        self.world_size = _env_int("WORLD_SIZE")
        self.node_rank = _env_int("NODE_RANK")
        self.cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    def __enter__(self) -> "TrainingRun":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.report("run_start", total_epochs=self.total_epochs, metadata=self.metadata)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.report("run_end", state="complete")
        else:
            self.report("run_end", state="failed", error=str(exc))

    def epoch_start(self, epoch: int, total_steps: Optional[int] = None, **metrics: Any) -> None:
        self.report("epoch_start", epoch=epoch, total_epochs=self.total_epochs, total_steps=total_steps, **metrics)

    def step(
        self,
        epoch: int,
        step: int,
        total_steps: Optional[int] = None,
        **metrics: Any,
    ) -> None:
        self.report("step", epoch=epoch, step=step, total_steps=total_steps, **metrics)

    def epoch_end(self, epoch: int, **metrics: Any) -> None:
        self.report("epoch_end", epoch=epoch, total_epochs=self.total_epochs, **metrics)

    def report(self, event: str, **payload: Any) -> None:
        record = {
            "time": time.time(),
            "event": event,
            "pid": self.pid,
            "run_name": self.run_name,
        }
        if self.gpu_index is not None:
            record["gpu_index"] = self.gpu_index
        if self.gpu_indices:
            record["gpu_indices"] = list(self.gpu_indices)
        if self.project:
            record["project"] = self.project
        if self.dataset:
            record["dataset"] = self.dataset
        if self.cuda_visible_devices:
            record["cuda_visible_devices"] = self.cuda_visible_devices
        if self.rank is not None:
            record["rank"] = self.rank
        if self.local_rank is not None:
            record["local_rank"] = self.local_rank
        if self.world_size is not None:
            record["world_size"] = self.world_size
        if self.node_rank is not None:
            record["node_rank"] = self.node_rank
        record.update({key: value for key, value in payload.items() if value is not None})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
