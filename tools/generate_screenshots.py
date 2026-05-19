from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.terminal_theme import TerminalTheme

from gpuwatch.models import GpuProcessSnapshot, GpuSnapshot, HostSnapshot, SystemSnapshot
from gpuwatch.render.rich_cli import render_dashboard
from gpuwatch.training.status import TrainingStatus


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "screenshots"

SOFT_DARK_TERMINAL_THEME = TerminalTheme(
    background=(17, 24, 39),
    foreground=(229, 231, 235),
    normal=[
        (15, 23, 42),
        (239, 68, 68),
        (34, 197, 94),
        (245, 158, 11),
        (56, 189, 248),
        (168, 85, 247),
        (45, 212, 191),
        (229, 231, 235),
    ],
    bright=[
        (71, 85, 105),
        (248, 113, 113),
        (74, 222, 128),
        (251, 191, 36),
        (125, 211, 252),
        (196, 181, 253),
        (103, 232, 249),
        (248, 250, 252),
    ],
)


CAPTURES = [
    ("01-wide-short-204x8", 204, 8, False, "soft-dark"),
    ("02-short-160x10", 160, 10, False, "soft-dark"),
    ("03-small-100x8", 100, 8, False, "soft-dark"),
    ("04-narrow-80x12", 80, 12, False, "soft-dark"),
    ("05-compact-100x18", 100, 18, False, "soft-dark"),
    ("06-medium-120x28", 120, 28, False, "soft-dark"),
    ("07-full-160x40", 160, 40, False, "soft-dark"),
    ("08-ascii-100x8", 100, 8, True, "terminal"),
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = demo_snapshot()
    statuses = demo_statuses()
    rows = []
    for name, width, height, ascii_only, theme in CAPTURES:
        path = OUTPUT_DIR / f"{name}.svg"
        render_capture(
            path=path,
            snapshot=snapshot,
            statuses=statuses,
            width=width,
            height=height,
            ascii_only=ascii_only,
            theme=theme,
        )
        rows.append((name, width, height, ascii_only, theme, path.name))
    write_gallery(rows)
    return 0


def render_capture(
    path: Path,
    snapshot: SystemSnapshot,
    statuses: list[TrainingStatus],
    width: int,
    height: int,
    ascii_only: bool,
    theme: str,
) -> None:
    console = Console(
        width=width,
        height=height,
        record=True,
        force_terminal=True,
        color_system="truecolor",
        file=StringIO(),
    )
    console.print(
        render_dashboard(
            snapshot,
            statuses,
            include_training=True,
            width=width,
            height=height,
            ascii_only=ascii_only,
            theme=theme,
        )
    )
    console.save_svg(path, title=f"gpuwatch {width}x{height}", theme=SOFT_DARK_TERMINAL_THEME)


def demo_snapshot() -> SystemSnapshot:
    processes = {
        3: (
            GpuProcessSnapshot(
                pid=271122,
                gpu_index=3,
                gpu_uuid="GPU-DEMO-03",
                gpu_memory_mb=2210,
                name="python",
                username="kmg",
                cmdline=("python", "train.py", "--run", "eth_seed7", "--gpu", "3"),
                cpu_percent=38.5,
                rss_mb=8192,
                elapsed_s=3722,
            ),
        ),
        4: (
            GpuProcessSnapshot(
                pid=271455,
                gpu_index=4,
                gpu_uuid="GPU-DEMO-04",
                gpu_memory_mb=2580,
                name="python",
                username="kmg",
                cmdline=("python", "train.py", "--run", "nba_seed1", "--gpu", "4"),
                cpu_percent=29.2,
                rss_mb=6144,
                elapsed_s=2844,
            ),
        ),
        5: (
            GpuProcessSnapshot(
                pid=271999,
                gpu_index=5,
                gpu_uuid="GPU-DEMO-05",
                gpu_memory_mb=14300,
                name="python",
                username="kmg",
                cmdline=("python", "train.py", "--run", "long_warn_job", "--gpu", "5"),
                cpu_percent=64.1,
                rss_mb=12288,
                elapsed_s=9120,
            ),
        ),
    }
    specs = [
        (0, "Tesla T4", 0, 3, 33, None, 9.0),
        (1, "Tesla T4", 0, 0, 35, None, 9.0),
        (2, "Tesla T4", 0, 0, 37, None, 9.0),
        (3, "Tesla T4", 95, 2200, 75, None, 51.0),
        (4, "Tesla T4", 49, 2600, 70, None, 36.5),
        (5, "Tesla T4", 24, 14360, 81, None, 65.2),
        (6, "Tesla T4", 0, 0, 38, None, 9.0),
        (7, "Tesla T4", 88, 7800, 84, None, 68.5),
    ]
    gpus = [
        GpuSnapshot(
            index=index,
            uuid=f"GPU-DEMO-{index:02d}",
            name=name,
            bus_id=f"00000000:{12 + index:02X}:00.0",
            temperature_c=temp,
            fan_percent=fan,
            utilization_gpu_percent=util,
            utilization_memory_percent=int(used / 15109 * 100),
            memory_used_mb=used,
            memory_total_mb=15109,
            power_draw_w=power,
            power_limit_w=70.0,
            processes=processes.get(index, ()),
        )
        for index, name, util, used, temp, fan, power in specs
    ]
    host = HostSnapshot(
        timestamp=1_779_174_060.0,
        hostname="cheetah-t4",
        cpu_percent=18.0,
        load_avg=(7.6, 7.6, 7.5),
        memory_percent=20.0,
        memory_used_mb=23900,
        memory_total_mb=125600,
    )
    return SystemSnapshot(host=host, gpus=tuple(gpus), errors=(), backend="nvml")


def demo_statuses() -> list[TrainingStatus]:
    return [
        TrainingStatus(
            pid=271122,
            gpu_index=3,
            run_name="eth_seed7",
            phase="train",
            epoch=4,
            max_epoch=9,
            step=64,
            total_steps=100,
            process_progress_percent=44.0,
            eta_seconds=7800,
            speed_per_second=8.2,
            speed_unit="step",
            loss=0.82,
            learning_rate=3e-4,
            age_seconds=3,
            state="running",
            state_reason="recent_heartbeat",
            confidence=0.95,
            evidence=("heartbeat",),
        ),
        TrainingStatus(
            pid=271455,
            gpu_index=4,
            run_name="nba_seed1",
            phase="eval",
            epoch=11,
            max_epoch=20,
            process_progress_percent=55.0,
            eta_seconds=4200,
            speed_per_second=5.1,
            speed_unit="step",
            loss=0.64,
            learning_rate=1e-4,
            age_seconds=11,
            state="running",
            state_reason="recent_heartbeat",
            confidence=0.82,
            evidence=("log-match",),
        ),
        TrainingStatus(
            pid=271999,
            gpu_index=5,
            run_name="long_warn_job",
            phase="train",
            epoch=8,
            max_epoch=12,
            process_progress_percent=67.0,
            eta_seconds=None,
            speed_per_second=0.0,
            speed_unit="step",
            loss=1.42,
            learning_rate=1e-5,
            age_seconds=520,
            state="stalled",
            state_reason="heartbeat_stale",
            confidence=0.74,
            evidence=("log", "stale"),
        ),
    ]


def write_gallery(rows: list[tuple[str, int, int, bool, str, str]]) -> None:
    lines = [
        "# gpuwatch responsive screenshots",
        "",
        "Generated with `python3 tools/generate_screenshots.py`.",
        "",
        "| Case | Size | Theme | ASCII | Preview |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for name, width, height, ascii_only, theme, filename in rows:
        lines.append(
            f"| `{name}` | `{width}x{height}` | `{theme}` | `{ascii_only}` | "
            f"![{name}]({filename}) |"
        )
    lines.append("")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
