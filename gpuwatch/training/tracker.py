from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


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


def _default_gpu_index() -> Optional[int]:
    explicit = _parse_gpu_index(os.environ.get("GPUWATCH_GPU_INDEX"))
    if explicit is not None:
        return explicit
    return _parse_gpu_index(os.environ.get("CUDA_VISIBLE_DEVICES"))


class TrainingRun:
    """Small heartbeat writer for training loops."""

    def __init__(
        self,
        run_name: str,
        total_epochs: Optional[int] = None,
        gpu_index: Optional[int] = None,
        skill_dir: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_name = run_name
        self.total_epochs = total_epochs
        self.gpu_index = gpu_index if gpu_index is not None else _default_gpu_index()
        self.metadata = metadata or {}
        base_dir = Path(skill_dir) if skill_dir else default_run_dir()
        self.run_dir = base_dir
        self.path = self.run_dir / f"run_{os.getpid()}.jsonl"
        self.pid = os.getpid()

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
        record.update({key: value for key, value in payload.items() if value is not None})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
