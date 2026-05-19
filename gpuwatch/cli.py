from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib import util as importlib_util
from typing import Iterable, List

from rich.console import Console

from gpuwatch.render.rich_cli import print_once, watch_plain
from gpuwatch.sampler import sample_once, watch
from gpuwatch.training.status import snapshot as training_snapshot


DEFAULT_INTERVAL = 0.25


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpuwatch", description="Responsive GPU monitor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    top = subparsers.add_parser("top", help="run the live monitor")
    _add_common(top)
    top.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    top.add_argument("--plain", action="store_true", help="use Rich live output instead of Textual")
    top.add_argument("--no-training", action="store_true", help="hide training progress panel")
    top.add_argument("--ascii", action="store_true", help="force ASCII separators")
    top.add_argument("--theme", choices=("soft-dark", "terminal", "light"), default="soft-dark")
    top.set_defaults(func=cmd_top)

    once = subparsers.add_parser("once", help="print one snapshot")
    _add_common(once)
    once.add_argument("--json", action="store_true", help="emit JSON")
    once.add_argument("--no-training", action="store_true", help="hide training progress panel")
    once.add_argument("--ascii", action="store_true", help="force ASCII separators")
    once.add_argument("--theme", choices=("soft-dark", "terminal", "light"), default="soft-dark")
    once.set_defaults(func=cmd_once)

    json_cmd = subparsers.add_parser("json", help="emit JSON snapshots")
    _add_common(json_cmd)
    json_cmd.add_argument("--watch", action="store_true", help="emit newline-delimited JSON")
    json_cmd.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    json_cmd.add_argument("--limit", type=int, default=None)
    json_cmd.set_defaults(func=cmd_json)

    train = subparsers.add_parser("train-status", help="scan training heartbeats and logs")
    train.add_argument("--root", action="append", default=[], help="project root to scan for logs")
    train.add_argument("--backend", choices=("auto", "fake"), default="auto")
    train.add_argument("--json", action="store_true")
    train.set_defaults(func=cmd_train_status)

    doctor = subparsers.add_parser("doctor", help="check runtime dependencies and NVML availability")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("auto", "fake"), default="auto")
    parser.add_argument("--root", action="append", default=[], help="project root for training log scan")


def cmd_top(args) -> int:
    roots = _roots(args.root)
    if args.plain:
        watch_plain(
            interval=args.interval,
            backend=args.backend,
            project_roots=roots,
            show_training=not args.no_training,
            ascii_only=args.ascii,
            theme=args.theme,
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
        )
    return 0


def cmd_once(args) -> int:
    snapshot = sample_once(backend=args.backend)
    statuses = (
        training_snapshot(project_roots=_roots(args.root), processes=list(snapshot.all_processes()))
        if not args.no_training
        else []
    )
    if args.json:
        payload = snapshot.to_dict()
        payload["training"] = [_status_to_dict(status) for status in statuses]
        print(json.dumps(payload, sort_keys=True))
    else:
        print_once(snapshot, statuses, include_training=not args.no_training, ascii_only=args.ascii, theme=args.theme)
    return 0


def cmd_json(args) -> int:
    if args.watch:
        for snapshot in watch(interval=args.interval, backend=args.backend, limit=args.limit):
            print(json.dumps(snapshot.to_dict(), sort_keys=True), flush=True)
    else:
        print(json.dumps(sample_once(backend=args.backend).to_dict(), sort_keys=True))
    return 0


def cmd_train_status(args) -> int:
    snapshot = sample_once(backend=args.backend)
    statuses = training_snapshot(project_roots=_roots(args.root), processes=list(snapshot.all_processes()))
    if args.json:
        print(json.dumps([_status_to_dict(status) for status in statuses], sort_keys=True))
        return 0
    console = Console()
    from rich.table import Table

    table = Table(title="Training Status", expand=True)
    for column in ("GPU", "PID", "Run", "State", "Phase", "Epoch", "Progress", "Metric", "Evidence"):
        table.add_column(column)
    for status in statuses:
        progress = "-" if status.process_progress_percent is None else f"{status.process_progress_percent:.1f}%"
        table.add_row(
            str(status.gpu_index) if status.gpu_index is not None else "-",
            str(status.pid) if status.pid is not None else "-",
            status.run_name or "-",
            status.state,
            status.phase,
            status.epoch_label(),
            progress,
            status.metric_label(),
            status.evidence_label(),
        )
    if not statuses:
        table.add_row("-", "-", "No training status", "-", "-", "-", "-", "-", "-")
    console.print(table)
    for error in snapshot.errors:
        console.print(f"[yellow]{error}[/yellow]")
    return 0


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
    snapshot = sample_once(backend="auto")
    if snapshot.errors:
        for error in snapshot.errors:
            console.print(f"[yellow]{error}[/yellow]")
        return 1
    console.print(f"[green]NVML OK: {len(snapshot.gpus)} GPU(s).[/green]")
    return 0


def _roots(values: Iterable[str]) -> List[str]:
    roots = list(values)
    env_roots = os.environ.get("GPUWATCH_PROJECT_ROOTS")
    if env_roots:
        roots.extend(item for item in env_roots.split(os.pathsep) if item)
    return roots


def _module_state(name: str) -> str:
    return "installed" if importlib_util.find_spec(name) is not None else "missing"


def _status_to_dict(status) -> dict:
    return {
        "pid": status.pid,
        "gpu_index": status.gpu_index,
        "run_name": status.run_name,
        "phase": status.phase,
        "epoch": status.epoch,
        "max_epoch": status.max_epoch,
        "step": status.step,
        "total_steps": status.total_steps,
        "progress_percent": status.process_progress_percent,
        "state": status.state,
        "confidence": status.confidence,
        "evidence": list(status.evidence),
        "log_path": status.log_path,
        "events_path": status.events_path,
    }


if __name__ == "__main__":
    raise SystemExit(main())
