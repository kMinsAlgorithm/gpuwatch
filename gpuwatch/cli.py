from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib import util as importlib_util
from pathlib import Path
from typing import Iterable, List

from rich.console import Console

from gpuwatch.render.rich_cli import print_once, watch_plain
from gpuwatch.sampler import sample_once, watch
from gpuwatch.training.status import snapshot as training_snapshot
from gpuwatch.view import ViewOptions, apply_view, parse_int_list, parse_str_list


DEFAULT_INTERVAL = 0.25
SCHEMA_VERSION = "0.2"
SORT_CHOICES = ("index", "util", "vram", "temp", "mem", "age", "progress", "eta", "run")
MODE_CHOICES = ("auto", "micro", "wide-short", "compact", "medium", "full")


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpuwatch", description="Responsive GPU monitor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    top = subparsers.add_parser("top", help="run the live monitor")
    _add_common(top)
    _add_view_options(top)
    _add_layout_options(top)
    top.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    top.add_argument("--plain", action="store_true", help="use Rich live output instead of Textual")
    top.add_argument("--no-training", action="store_true", help="hide training progress panel")
    top.add_argument("--ascii", action="store_true", help="force ASCII separators")
    top.add_argument("--theme", choices=("soft-dark", "terminal", "light"), default="soft-dark")
    top.set_defaults(func=cmd_top)

    once = subparsers.add_parser("once", help="print one snapshot")
    _add_common(once)
    _add_view_options(once)
    _add_layout_options(once)
    once.add_argument("--json", action="store_true", help="emit JSON")
    once.add_argument("--no-training", action="store_true", help="hide training progress panel")
    once.add_argument("--ascii", action="store_true", help="force ASCII separators")
    once.add_argument("--theme", choices=("soft-dark", "terminal", "light"), default="soft-dark")
    once.set_defaults(func=cmd_once)

    json_cmd = subparsers.add_parser("json", help="emit JSON snapshots")
    _add_common(json_cmd)
    _add_view_options(json_cmd)
    json_cmd.add_argument("--watch", action="store_true", help="emit newline-delimited JSON")
    json_cmd.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    json_cmd.add_argument("--limit", type=int, default=None)
    json_cmd.add_argument("--no-training", action="store_true", help="omit training statuses")
    json_cmd.add_argument("--schema-version", action="store_true", help="print the JSON schema version and exit")
    json_cmd.set_defaults(func=cmd_json)

    train = subparsers.add_parser("train-status", help="scan training heartbeats and logs")
    _add_view_options(train)
    train.add_argument("--root", action="append", default=[], help="project root to scan for logs")
    train.add_argument("--backend", choices=("auto", "fake"), default="auto")
    train.add_argument("--json", action="store_true")
    train.add_argument("--watch", action="store_true")
    train.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    train.add_argument("--stale-after", type=float, default=900.0)
    train.add_argument("--failed-only", action="store_true")
    train.add_argument("--run", default=None)
    train.add_argument("--explain", action="store_true")
    train.set_defaults(func=cmd_train_status)

    doctor = subparsers.add_parser("doctor", help="check runtime dependencies and NVML availability")
    doctor.add_argument("--backend", choices=("auto", "fake"), default="auto")
    doctor.add_argument("--root", action="append", default=[])
    doctor.set_defaults(func=cmd_doctor)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("auto", "fake"), default="auto")
    parser.add_argument("--root", action="append", default=[], help="project root for training log scan")


def _add_layout_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=MODE_CHOICES, default="auto")
    parser.add_argument("--explain-layout", action="store_true")


def _add_view_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", default=None, help="GPU index list/range, e.g. 0,2-3")
    parser.add_argument("--pid", default=None, help="PID list/range")
    parser.add_argument("--user", default=None, help="filter process user substring")
    parser.add_argument("--cmd", default=None, help="filter process command substring")
    parser.add_argument("--state", default=None, help="training state list")
    parser.add_argument("--sort", choices=SORT_CHOICES, default="index")
    parser.add_argument("--reverse", action="store_true")


def cmd_top(args) -> int:
    roots = _roots(args.root)
    view_options = _view_options(args)
    if args.plain:
        watch_plain(
            interval=args.interval,
            backend=args.backend,
            project_roots=roots,
            show_training=not args.no_training,
            ascii_only=args.ascii,
            theme=args.theme,
            display_mode=args.mode,
            explain_layout=args.explain_layout,
            view_options=view_options,
        )
        return 0
    try:
        from gpuwatch.render.textual_app import run_textual

        run_textual(
            interval=args.interval,
            backend=args.backend,
            project_roots=roots,
            show_training=not args.no_training,
            ascii_only=args.ascii,
            theme=args.theme,
            display_mode=args.mode,
            explain_layout=args.explain_layout,
            view_options=view_options,
        )
    except RuntimeError as exc:
        console = Console(stderr=True)
        console.print(f"[yellow]{exc} Falling back to --plain.[/yellow]")
        watch_plain(
            interval=args.interval,
            backend=args.backend,
            project_roots=roots,
            show_training=not args.no_training,
            ascii_only=args.ascii,
            theme=args.theme,
            display_mode=args.mode,
            explain_layout=args.explain_layout,
            view_options=view_options,
        )
    return 0


def cmd_once(args) -> int:
    snapshot = sample_once(backend=args.backend)
    statuses = (
        training_snapshot(project_roots=_roots(args.root), processes=list(snapshot.all_processes()))
        if not args.no_training
        else []
    )
    snapshot, statuses = apply_view(snapshot, statuses, _view_options(args))
    if args.json:
        payload = _snapshot_payload(snapshot, statuses, include_training=not args.no_training)
        print(json.dumps(payload, sort_keys=True))
    else:
        print_once(
            snapshot,
            statuses,
            include_training=not args.no_training,
            ascii_only=args.ascii,
            theme=args.theme,
            display_mode=args.mode,
            explain_layout=args.explain_layout,
        )
    return 0


def cmd_json(args) -> int:
    if args.schema_version:
        print(SCHEMA_VERSION)
        return 0
    view_options = _view_options(args)
    if args.watch:
        for snapshot in watch(interval=args.interval, backend=args.backend, limit=args.limit):
            statuses = (
                training_snapshot(project_roots=_roots(args.root), processes=list(snapshot.all_processes()))
                if not args.no_training
                else []
            )
            snapshot, statuses = apply_view(snapshot, statuses, view_options)
            print(json.dumps(_snapshot_payload(snapshot, statuses, include_training=not args.no_training), sort_keys=True), flush=True)
    else:
        snapshot = sample_once(backend=args.backend)
        statuses = (
            training_snapshot(project_roots=_roots(args.root), processes=list(snapshot.all_processes()))
            if not args.no_training
            else []
        )
        snapshot, statuses = apply_view(snapshot, statuses, view_options)
        print(json.dumps(_snapshot_payload(snapshot, statuses, include_training=not args.no_training), sort_keys=True))
    return 0


def cmd_train_status(args) -> int:
    def emit_once() -> None:
        snapshot = sample_once(backend=args.backend)
        statuses = training_snapshot(
            project_roots=_roots(args.root),
            processes=list(snapshot.all_processes()),
            stale_after=args.stale_after,
        )
        snapshot_view, statuses_view = apply_view(snapshot, statuses, _view_options(args))
        if args.json:
            print(json.dumps([_status_to_dict(status) for status in statuses_view], sort_keys=True), flush=True)
        else:
            _print_train_status_table(statuses_view, snapshot_view.errors, explain=args.explain)

    if args.watch:
        while True:
            emit_once()
            time.sleep(args.interval)
        return 0
    emit_once()
    return 0


def _print_train_status_table(statuses, errors, explain: bool = False) -> None:
    console = Console()
    from rich.table import Table

    table = Table(title="Training Status", expand=True)
    columns = ["GPU", "PID", "Run", "State", "Phase", "Epoch", "Progress", "ETA", "Speed", "HB", "Metric", "Evidence"]
    if explain:
        columns.append("Reason")
    for column in columns:
        table.add_column(column)
    for status in statuses:
        progress = "-" if status.process_progress_percent is None else f"{status.process_progress_percent:.1f}%"
        row = [
            str(status.gpu_index) if status.gpu_index is not None else "-",
            str(status.pid) if status.pid is not None else "-",
            status.run_name or "-",
            status.state,
            status.phase,
            status.epoch_label(),
            progress,
            _duration_label(status.eta_seconds),
            _status_speed_label(status),
            _duration_label(status.age_seconds if status.age_seconds is not None else status.stale_seconds),
            status.metric_label(),
            status.evidence_label(),
        ]
        if explain:
            row.append(status.state_reason or "-")
        table.add_row(*row)
    if not statuses:
        empty = ["-", "-", "No training status", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
        if explain:
            empty.append("-")
        table.add_row(*empty)
    console.print(table)
    for error in errors:
        console.print(f"[yellow]{error}[/yellow]")


def cmd_doctor(args) -> int:
    console = Console()
    rows = [
        ("python", sys.version.split()[0], "ok"),
        ("rich", _module_state("rich"), "required"),
        ("psutil", _module_state("psutil"), "required"),
        ("pynvml", _module_state("pynvml"), "provided by nvidia-ml-py"),
        ("textual", _module_state("textual"), "optional for full TUI"),
    ]
    from rich.table import Table

    table = Table(title="gpuwatch doctor")
    table.add_column("Item")
    table.add_column("State")
    table.add_column("Note")
    for item, state, note in rows:
        table.add_row(item, state, note)
    console.print(table)
    snapshot = sample_once(backend=args.backend)
    exit_code = 0
    if snapshot.errors:
        for error in snapshot.errors:
            console.print(f"[yellow]{error}[/yellow]")
        exit_code = 1
    else:
        console.print(f"[green]{snapshot.backend.upper()} OK: {len(snapshot.gpus)} GPU(s).[/green]")

    roots = _roots(args.root)
    if roots:
        root_table = Table(title="Project Roots")
        root_table.add_column("Root")
        root_table.add_column("State")
        root_table.add_column("Logs")
        root_table.add_column("Heartbeats")
        root_table.add_column("Statuses")
        root_table.add_column("Stale/Orphaned")
        for root in roots:
            root_path = Path(root).expanduser()
            if not root_path.exists():
                root_table.add_row(str(root_path), "missing", "-", "-", "-", "-")
                exit_code = 1
                continue
            logs = _bounded_count(root_path, "*.log")
            heartbeats = _bounded_count(root_path, "run_*.jsonl")
            statuses = training_snapshot(project_roots=[str(root_path)], processes=[])
            troubled = sum(1 for status in statuses if status.state in ("stalled", "orphaned", "failed"))
            root_table.add_row(str(root_path), "ok", str(logs), str(heartbeats), str(len(statuses)), str(troubled))
        console.print(root_table)
    return exit_code


def _roots(values: Iterable[str]) -> List[str]:
    roots = list(values)
    env_roots = os.environ.get("GPUWATCH_PROJECT_ROOTS")
    if env_roots:
        roots.extend(item for item in env_roots.split(os.pathsep) if item)
    return roots


def _module_state(name: str) -> str:
    return "installed" if importlib_util.find_spec(name) is not None else "missing"


def _bounded_count(root: Path, pattern: str, limit: int = 500) -> int:
    count = 0
    try:
        for _path in root.rglob(pattern):
            count += 1
            if count >= limit:
                break
    except OSError:
        return 0
    return count


def _view_options(args) -> ViewOptions:
    try:
        return ViewOptions(
            gpu_indices=parse_int_list(getattr(args, "gpu", None)),
            pids=parse_int_list(getattr(args, "pid", None)),
            user=getattr(args, "user", None),
            cmd=getattr(args, "cmd", None),
            states=parse_str_list(getattr(args, "state", None)),
            run=getattr(args, "run", None),
            failed_only=bool(getattr(args, "failed_only", False)),
            sort=getattr(args, "sort", "index"),
            reverse=bool(getattr(args, "reverse", False)),
        )
    except ValueError as exc:
        raise SystemExit(f"invalid filter value: {exc}") from exc


def _snapshot_payload(snapshot, statuses, include_training: bool = True) -> dict:
    payload = snapshot.to_dict()
    payload["schema_version"] = SCHEMA_VERSION
    if include_training:
        payload["training"] = [_status_to_dict(status) for status in statuses]
    return payload


def _duration_label(value) -> str:
    from gpuwatch.render.formatters import seconds

    return "-" if value is None else seconds(value)


def _status_speed_label(status) -> str:
    if status.speed_per_second is None:
        return "-"
    unit = status.speed_unit or "step"
    suffix = "it/s" if unit == "step" else f"{unit}/s"
    return f"{status.speed_per_second:.2g} {suffix}"


def _status_to_dict(status) -> dict:
    return {
        "pid": status.pid,
        "gpu_index": status.gpu_index,
        "run_name": status.run_name,
        "project": status.project,
        "phase": status.phase,
        "epoch": status.epoch,
        "max_epoch": status.max_epoch,
        "step": status.step,
        "total_steps": status.total_steps,
        "progress_percent": status.process_progress_percent,
        "eta_seconds": status.eta_seconds,
        "speed_per_second": status.speed_per_second,
        "speed_unit": status.speed_unit,
        "loss": status.loss,
        "learning_rate": status.learning_rate,
        "checkpoint_path": status.checkpoint_path,
        "rank": status.rank,
        "local_rank": status.local_rank,
        "world_size": status.world_size,
        "node_rank": status.node_rank,
        "metric_name": status.metric_name,
        "metric_value": status.metric_value,
        "best_epoch": status.best_epoch,
        "val_ade": status.val_ade,
        "val_fde": status.val_fde,
        "score": status.score,
        "state": status.state,
        "state_reason": status.state_reason,
        "last_update_time": status.last_update_time,
        "age_seconds": status.age_seconds,
        "stale_seconds": status.stale_seconds,
        "confidence": status.confidence,
        "evidence": list(status.evidence),
        "log_path": status.log_path,
        "events_path": status.events_path,
    }


if __name__ == "__main__":
    raise SystemExit(main())
