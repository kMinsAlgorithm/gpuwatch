from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gpuwatch.models import GpuProcessSnapshot
from gpuwatch.training.tracker import default_run_dir


INFO_EPOCH_RE = re.compile(r"\[INFO\](?:\[[^\]]+\]){0,5}\[Epoch\s+(\d+)\s*/\s*(\d+)\]")
INLINE_EPOCH_RE = re.compile(r"Epochs?:\s*(\d+)\s*/\s*(\d+)")
ITER_RE = re.compile(r"It:\s*(\d+)\s*/\s*(\d+)")
PHASE_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]\[(TRAIN|Train|TEST|Test|VAL|Val)\]\s+Epoch\s+(\d+)")
RUN_NAME_RE = re.compile(r"\[INFO\]\[([^\]]+)\]\[Epoch")
METRIC_RE = re.compile(r"(?:val_)?(ADE|FDE|score|best_epoch)\s*[:=]\s*([0-9.]+)", re.IGNORECASE)
BEST_RE = re.compile(r"\[BEST\].*epoch\s*=\s*(\d+).*ADE\s*=\s*([0-9.]+).*FDE\s*=\s*([0-9.]+)", re.IGNORECASE)
_LOG_CACHE: Dict[Tuple[Tuple[str, ...], int, float], Tuple[float, List["TrainingStatus"]]] = {}
_LOG_CACHE_TTL_SECONDS = 15.0


@dataclass(frozen=True)
class TrainingStatus:
    pid: Optional[int] = None
    gpu_index: Optional[int] = None
    run_name: Optional[str] = None
    phase: str = "unknown"
    epoch: Optional[int] = None
    max_epoch: Optional[int] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    process_progress_percent: Optional[float] = None
    best_epoch: Optional[int] = None
    val_ade: Optional[float] = None
    val_fde: Optional[float] = None
    score: Optional[float] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    checkpoint_path: Optional[str] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    world_size: Optional[int] = None
    node_rank: Optional[int] = None
    project: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    last_update_time: Optional[float] = None
    age_seconds: Optional[float] = None
    state_reason: Optional[str] = None
    eta_seconds: Optional[float] = None
    speed_per_second: Optional[float] = None
    speed_unit: Optional[str] = None
    state: str = "unknown"
    stale_seconds: Optional[float] = None
    confidence: float = 0.0
    evidence: Tuple[str, ...] = field(default_factory=tuple)
    log_path: Optional[str] = None
    events_path: Optional[str] = None

    def compact_label(self) -> str:
        pieces = [self.phase, self.epoch_label()]
        if self.process_progress_percent is not None:
            pieces.append(f"{self.process_progress_percent:.0f}%")
        return " ".join(piece for piece in pieces if piece and piece != "-")

    def epoch_label(self) -> str:
        if self.epoch is None:
            return "-"
        if self.max_epoch is None:
            return str(self.epoch)
        return f"{self.epoch}/{self.max_epoch}"

    def metric_label(self) -> str:
        parts = []
        if self.val_ade is not None:
            parts.append(f"ADE {self.val_ade:.4f}")
        if self.val_fde is not None:
            parts.append(f"FDE {self.val_fde:.4f}")
        if self.score is not None:
            parts.append(f"score {self.score:.4f}")
        if self.metric_name and self.metric_value is not None:
            parts.append(f"{self.metric_name} {self.metric_value:.4g}")
        if self.loss is not None:
            parts.append(f"loss {self.loss:.4g}")
        if self.learning_rate is not None:
            parts.append(f"lr {self.learning_rate:.3g}")
        if self.best_epoch is not None:
            parts.append(f"best {self.best_epoch}")
        return " | ".join(parts) if parts else "-"

    def evidence_label(self) -> str:
        return ",".join(self.evidence[:3]) if self.evidence else "-"


def snapshot(
    project_roots: Iterable[str],
    processes: Sequence[GpuProcessSnapshot] = (),
    max_logs: int = 80,
    stale_after: float = 900.0,
) -> List[TrainingStatus]:
    heartbeat_statuses = _heartbeat_statuses(processes, stale_after=stale_after)
    by_pid: Dict[int, TrainingStatus] = {
        status.pid: status for status in heartbeat_statuses if status.pid is not None
    }

    log_statuses = _cached_log_statuses(project_roots, max_logs=max_logs, stale_after=stale_after)
    if processes:
        bound = []
        bound_pids = set()
        for process in processes:
            if process.pid in by_pid:
                bound.append(_attach_process(by_pid[process.pid], process, 0.95, "heartbeat"))
                bound_pids.add(process.pid)
                continue
            log_status = _best_log_match(process, log_statuses)
            if log_status is not None:
                bound.append(_attach_process(log_status, process, min(0.82, log_status.confidence), "log-match"))
            else:
                bound.append(
                    TrainingStatus(
                        pid=process.pid,
                        gpu_index=process.gpu_index,
                        run_name=_infer_run_name(process.cmdline),
                        phase="unknown",
                        state="unbound",
                        state_reason="gpu_process_only",
                        confidence=0.25,
                        evidence=("gpu-process",),
                    )
                )
        detached = [status for status in heartbeat_statuses if status.pid not in bound_pids]
        return bound + detached

    statuses = list(heartbeat_statuses)
    statuses.extend(log_statuses[:max_logs])
    return statuses


def parse_log(path: Path, stale_after: float = 900.0) -> TrainingStatus:
    text = _tail_text(path)
    mtime = path.stat().st_mtime
    stale = max(0.0, time.time() - mtime)
    run_name = path.stem
    phase = "unknown"
    epoch = None
    max_epoch = None
    step = None
    total_steps = None
    metrics: Dict[str, float] = {}
    evidence = ["log"]

    for line in text.splitlines():
        run_match = RUN_NAME_RE.search(line)
        if run_match:
            run_name = run_match.group(1)

        phase_match = PHASE_RE.search(line)
        if phase_match:
            raw_phase = phase_match.group(2).lower()
            phase = "eval" if raw_phase in ("test", "val") else "train"
            epoch = int(phase_match.group(3))

        info_match = INFO_EPOCH_RE.search(line) or INLINE_EPOCH_RE.search(line)
        if info_match:
            epoch = int(info_match.group(1))
            max_epoch = int(info_match.group(2))
            phase = "between_epochs"
            evidence.append("epoch-regex")

        iter_match = ITER_RE.search(line)
        if iter_match:
            step = int(iter_match.group(1))
            total_steps = int(iter_match.group(2))
            phase = "train"
            evidence.append("iter-regex")

        best_match = BEST_RE.search(line)
        if best_match:
            metrics["best_epoch"] = float(best_match.group(1))
            metrics["ade"] = float(best_match.group(2))
            metrics["fde"] = float(best_match.group(3))

        for key, value in METRIC_RE.findall(line):
            key_lower = key.lower()
            metrics[key_lower] = float(value)

        if "Training Finished" in line or "run_end" in line:
            phase = "complete"

    progress = _progress_percent(epoch, max_epoch, step, total_steps, phase)
    state = "stalled" if stale > stale_after and phase not in ("complete",) else "running"
    if phase == "complete":
        state = "complete"
    return TrainingStatus(
        run_name=run_name,
        phase=phase,
        epoch=epoch,
        max_epoch=max_epoch,
        step=step,
        total_steps=total_steps,
        process_progress_percent=progress,
        best_epoch=int(metrics["best_epoch"]) if "best_epoch" in metrics else None,
        val_ade=metrics.get("ade"),
        val_fde=metrics.get("fde"),
        score=metrics.get("score"),
        state=state,
        state_reason="heartbeat_stale" if state == "stalled" else ("run_end_complete" if state == "complete" else "log_match"),
        last_update_time=mtime,
        age_seconds=stale,
        stale_seconds=stale,
        confidence=0.65 if epoch is not None else 0.35,
        evidence=tuple(dict.fromkeys(evidence)),
        log_path=str(path),
    )


def _log_statuses(project_roots: Iterable[str], max_logs: int, stale_after: float) -> List[TrainingStatus]:
    logs = []
    for root in project_roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        logs.extend(_recent_files(root_path, "*.log", max_logs=max_logs))
    statuses = []
    for path in logs[:max_logs]:
        try:
            statuses.append(parse_log(path, stale_after=stale_after))
        except (OSError, UnicodeError):
            continue
    statuses.sort(key=lambda item: item.stale_seconds if item.stale_seconds is not None else 1e18)
    return statuses


def _cached_log_statuses(project_roots: Iterable[str], max_logs: int, stale_after: float) -> List[TrainingStatus]:
    roots = tuple(str(Path(root).expanduser()) for root in project_roots)
    key = (roots, max_logs, float(stale_after))
    now = time.time()
    cached = _LOG_CACHE.get(key)
    if cached is not None and now - cached[0] < _LOG_CACHE_TTL_SECONDS:
        return cached[1]
    statuses = _log_statuses(roots, max_logs=max_logs, stale_after=stale_after)
    _LOG_CACHE[key] = (now, statuses)
    return statuses


def _heartbeat_statuses(processes: Sequence[GpuProcessSnapshot], stale_after: float) -> List[TrainingStatus]:
    del processes
    statuses = []
    for run_file in _heartbeat_files():
        try:
            status = _parse_heartbeat(run_file, stale_after=stale_after)
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        statuses.append(status)
    return statuses


def _heartbeat_files() -> List[Path]:
    roots = [default_run_dir()]
    env_extra = os.environ.get("GPUWATCH_EXTRA_RUN_DIRS")
    if env_extra:
        roots.extend(Path(item).expanduser() for item in env_extra.split(os.pathsep) if item)
    files = []
    for root in roots:
        if root.exists():
            files.extend(_recent_files(root, "run_*.jsonl", max_logs=200))
    return files


def _parse_heartbeat(path: Path, stale_after: float = 900.0) -> TrainingStatus:
    records = []
    start = {}
    for line in _tail_text(path).splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(record)
        if record.get("event") == "run_start":
            start = record
    latest = records[-1] if records else None
    if latest is None:
        raise json.JSONDecodeError("empty heartbeat", "", 0)

    event = latest.get("event", "unknown")
    pid = _optional_int(_first_present(latest.get("pid"), start.get("pid")))
    epoch = _optional_int(latest.get("epoch"))
    max_epoch = _optional_int(latest.get("total_epochs") or start.get("total_epochs"))
    step = _optional_int(latest.get("step"))
    total_steps = _optional_int(latest.get("total_steps"))
    phase = {
        "run_start": "starting",
        "epoch_start": "train",
        "step": "train",
        "epoch_end": "between_epochs",
        "run_end": "complete" if latest.get("state") == "complete" else "failed",
    }.get(event, event)
    progress = _progress_percent(epoch, max_epoch, step, total_steps, phase)
    last_time = float(latest.get("time", path.stat().st_mtime))
    age = max(0.0, time.time() - last_time)
    state = "running"
    state_reason = "recent_heartbeat"
    if phase == "complete":
        state = "complete"
        state_reason = "run_end_complete"
    elif phase == "failed":
        state = "failed"
        state_reason = "run_end_failed"
    elif age > stale_after:
        state = "stalled"
        state_reason = "heartbeat_stale"
    if pid is not None and phase not in ("complete", "failed") and not _pid_exists(pid):
        state = "orphaned"
        state_reason = "pid_missing"
    speed, speed_unit, eta = _heartbeat_speed_eta(records, start, latest, phase)
    metric_name = latest.get("metric_name") or start.get("metric_name")
    return TrainingStatus(
        pid=pid,
        gpu_index=_optional_int(
            _first_present(
                latest.get("gpu_index"),
                start.get("gpu_index"),
                _metadata_value(latest, "gpu_index"),
                _metadata_value(start, "gpu_index"),
            )
        ),
        project=_first_present(latest.get("project"), start.get("project"), _metadata_value(latest, "project"), _metadata_value(start, "project")),
        run_name=latest.get("run_name") or start.get("run_name") or path.stem,
        phase=phase,
        epoch=epoch,
        max_epoch=max_epoch,
        step=step,
        total_steps=total_steps,
        process_progress_percent=progress,
        val_ade=_optional_float(latest.get("val_ade") or latest.get("ade")),
        val_fde=_optional_float(latest.get("val_fde") or latest.get("fde")),
        score=_optional_float(latest.get("score")),
        best_epoch=_optional_int(latest.get("best_epoch")),
        loss=_optional_float(_first_present(latest.get("loss"), latest.get("train_loss"), latest.get("val_loss"))),
        learning_rate=_optional_float(_first_present(latest.get("learning_rate"), latest.get("lr"))),
        checkpoint_path=_first_present(latest.get("checkpoint_path"), latest.get("checkpoint")),
        metric_name=metric_name,
        metric_value=_optional_float(_first_present(latest.get("metric_value"), latest.get(metric_name) if metric_name else None)),
        rank=_optional_int(_first_present(latest.get("rank"), start.get("rank"))),
        local_rank=_optional_int(_first_present(latest.get("local_rank"), start.get("local_rank"))),
        world_size=_optional_int(_first_present(latest.get("world_size"), start.get("world_size"))),
        node_rank=_optional_int(_first_present(latest.get("node_rank"), start.get("node_rank"))),
        state=state,
        state_reason=state_reason,
        eta_seconds=eta,
        speed_per_second=speed,
        speed_unit=speed_unit,
        last_update_time=last_time,
        age_seconds=age,
        stale_seconds=age,
        confidence=0.95,
        evidence=("heartbeat",),
        events_path=str(path),
    )


def _attach_process(
    status: TrainingStatus,
    process: GpuProcessSnapshot,
    confidence: float,
    evidence: str,
) -> TrainingStatus:
    merged_evidence = tuple(dict.fromkeys((evidence,) + status.evidence))
    return TrainingStatus(
        pid=process.pid,
        gpu_index=process.gpu_index,
        run_name=status.run_name or _infer_run_name(process.cmdline),
        phase=status.phase,
        epoch=status.epoch,
        max_epoch=status.max_epoch,
        step=status.step,
        total_steps=status.total_steps,
        process_progress_percent=status.process_progress_percent,
        best_epoch=status.best_epoch,
        val_ade=status.val_ade,
        val_fde=status.val_fde,
        score=status.score,
        loss=status.loss,
        learning_rate=status.learning_rate,
        checkpoint_path=status.checkpoint_path,
        rank=status.rank,
        local_rank=status.local_rank,
        world_size=status.world_size,
        node_rank=status.node_rank,
        project=status.project,
        metric_name=status.metric_name,
        metric_value=status.metric_value,
        last_update_time=status.last_update_time,
        age_seconds=status.age_seconds,
        state_reason=evidence if status.state_reason is None else status.state_reason,
        eta_seconds=status.eta_seconds,
        speed_per_second=status.speed_per_second,
        speed_unit=status.speed_unit,
        state=status.state,
        stale_seconds=status.stale_seconds,
        confidence=confidence,
        evidence=merged_evidence,
        log_path=status.log_path,
        events_path=status.events_path,
    )


def _best_log_match(
    process: GpuProcessSnapshot,
    statuses: Sequence[TrainingStatus],
) -> Optional[TrainingStatus]:
    command = " ".join(process.cmdline).lower()
    inferred = (_infer_run_name(process.cmdline) or "").lower()
    best = None
    best_score = 0
    for status in statuses:
        candidates = [status.run_name or "", Path(status.log_path or "").stem]
        score = 0
        for candidate in candidates:
            candidate_lower = candidate.lower()
            if candidate_lower and candidate_lower in command:
                score += 3
            if inferred and candidate_lower and inferred in candidate_lower:
                score += 2
        if status.stale_seconds is not None and status.stale_seconds < 3600:
            score += 1
        if score > best_score:
            best = status
            best_score = score
    return best if best_score > 0 else None


def _infer_run_name(cmdline: Sequence[str]) -> Optional[str]:
    for idx, token in enumerate(cmdline):
        if token in ("--run-name", "--run_name", "--name") and idx + 1 < len(cmdline):
            return cmdline[idx + 1]
    for token in reversed(cmdline):
        if token.endswith(".py"):
            return Path(token).stem
    return None


def _progress_percent(
    epoch: Optional[int],
    max_epoch: Optional[int],
    step: Optional[int],
    total_steps: Optional[int],
    phase: str,
) -> Optional[float]:
    if max_epoch in (None, 0) or epoch is None:
        return None
    phase_fraction = {
        "starting": 0.0,
        "train": 0.35,
        "eval": 0.9,
        "between_epochs": 1.0,
        "complete": 1.0,
    }.get(phase, 0.0)
    if total_steps and step is not None:
        phase_fraction = max(0.0, min(0.8, float(step) / float(total_steps) * 0.8))
    progress = (float(epoch) + phase_fraction) / float(max_epoch)
    return max(0.0, min(100.0, progress * 100.0))


def _heartbeat_speed_eta(
    records: Sequence[dict],
    start: dict,
    latest: dict,
    phase: str,
) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    latest_position = _heartbeat_position(latest, start)
    if latest_position is None:
        return None, None, None
    latest_time = _optional_float(latest.get("time"))
    if latest_time is None:
        return None, None, None
    latest_pos, latest_total, latest_unit = latest_position
    previous = None
    for record in reversed(records[:-1]):
        position = _heartbeat_position(record, start)
        record_time = _optional_float(record.get("time"))
        if position is None or record_time is None:
            continue
        pos, total, unit = position
        if unit != latest_unit or total != latest_total:
            continue
        if record_time < latest_time and pos < latest_pos:
            previous = (record_time, pos)
            break
    if previous is None:
        return None, latest_unit, None
    previous_time, previous_pos = previous
    elapsed = latest_time - previous_time
    advanced = latest_pos - previous_pos
    if elapsed <= 0 or advanced <= 0:
        return None, latest_unit, None
    speed = advanced / elapsed
    eta = None
    if phase not in ("complete", "failed") and speed > 0:
        eta = max(0.0, (latest_total - latest_pos) / speed)
    return speed, latest_unit, eta


def _heartbeat_position(record: dict, start: dict) -> Optional[Tuple[float, float, str]]:
    epoch = _optional_int(record.get("epoch"))
    max_epoch = _optional_int(record.get("total_epochs") or start.get("total_epochs"))
    step = _optional_int(record.get("step"))
    total_steps = _optional_int(record.get("total_steps"))
    if epoch is not None and max_epoch and step is not None and total_steps:
        total = float(max_epoch * total_steps)
        position = float(epoch * total_steps + step)
        return max(0.0, min(total, position)), total, "step"
    if epoch is not None and max_epoch:
        return float(epoch), float(max_epoch), "epoch"
    return None


def _recent_files(root: Path, pattern: str, max_logs: int) -> List[Path]:
    files = []
    for path in root.rglob(pattern):
        try:
            files.append((path.stat().st_mtime, path))
        except OSError:
            continue
    files.sort(reverse=True, key=lambda item: item[0])
    return [path for _, path in files[:max_logs]]


def _tail_text(path: Path, max_bytes: int = 262144) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _metadata_value(record, key: str):
    metadata = record.get("metadata") if isinstance(record, dict) else None
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _pid_exists(pid: Optional[int]) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True


def _optional_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
