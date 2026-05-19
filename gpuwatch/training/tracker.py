from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def default_run_dir() -> Path:
    return Path(os.environ.get("GPUWATCH_RUN_DIR", Path.home() / ".cache" / "gpuwatch" / "runs"))


class TrainingRun:
    """Small heartbeat writer for training loops."""

    def __init__(
        self,
        run_name: str,
        total_epochs: Optional[int] = None,
        skill_dir: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_name = run_name
        self.total_epochs = total_epochs
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
        record.update({key: value for key, value in payload.items() if value is not None})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
