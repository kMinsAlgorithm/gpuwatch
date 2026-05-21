from __future__ import annotations

from dataclasses import dataclass
import json
import time
import math
from typing import Iterable, List, Optional

from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gpuwatch.models import GpuProcessSnapshot, SystemSnapshot
from gpuwatch.render.formatters import bar, clamp_cells, clamp_text, mb, na, percent, seconds
from gpuwatch.sampler import sample_once
from gpuwatch.training.status import TrainingStatus, snapshot as training_snapshot
from gpuwatch.view import ViewOptions, apply_view


@dataclass(frozen=True)
class RenderGlyphs:
    tile_left: str
    tile_right: str
    section: str
    overflow: str
    marker: str
    top_left: str
    top_right: str
    bottom_left: str
    bottom_right: str
    horizontal: str
    vertical: str


@dataclass(frozen=True)
class RenderTheme:
    name: str
    background: str
    surface: str
    surface_alt: str
    border: str
    text: str
    muted: str
    warn: str
    crit: str
    ok: str
    info: str

    @property
    def bg_style(self) -> str:
        return _style(self.text, self.background)

    @property
    def surface_style(self) -> str:
        return _style(self.text, self.surface)

    @property
    def alt_style(self) -> str:
        return _style(self.text, self.surface_alt)

    @property
    def muted_style(self) -> str:
        return _style(self.muted, self.background)

    @property
    def border_style(self) -> str:
        return _style(self.border, self.surface)


@dataclass(frozen=True)
class TileSpec:
    target_width: int = 32
    min_width: int = 26
    max_width: int = 38
    height: int = 3
    gap_width: int = 1


@dataclass(frozen=True)
class GpuHealth:
    label: str
    priority: int
    style_key: str
    reason: str = ""


@dataclass(frozen=True)
class LayoutDecision:
    mode: str
    source: str
    reason: str
    width: Optional[int]
    height: Optional[int]


UNICODE_GLYPHS = RenderGlyphs("┃", "┃", "│", "…", "…", "┌", "┐", "└", "┘", "─", "│")
ASCII_GLYPHS = RenderGlyphs("|", "|", "|", "~", "~", "+", "+", "+", "+", "-", "|")
GPU_BADGE_STYLES = ("cyan", "magenta", "green", "yellow", "blue", "bright_cyan")
TILE_SPEC = TileSpec()
THEMES = {
    "soft-dark": RenderTheme(
        name="soft-dark",
        background="#111827",
        surface="#1f2937",
        surface_alt="#243244",
        border="#475569",
        text="#e5e7eb",
        muted="#94a3b8",
        warn="#f59e0b",
        crit="#ef4444",
        ok="#22c55e",
        info="#38bdf8",
    ),
    "terminal": RenderTheme(
        name="terminal",
        background="",
        surface="",
        surface_alt="",
        border="dim",
        text="",
        muted="dim",
        warn="yellow",
        crit="red",
        ok="green",
        info="cyan",
    ),
    "light": RenderTheme(
        name="light",
        background="#f3f4f6",
        surface="#ffffff",
        surface_alt="#e5e7eb",
        border="#94a3b8",
        text="#111827",
        muted="#475569",
        warn="#b45309",
        crit="#b91c1c",
        ok="#15803d",
        info="#0369a1",
    ),
}


def print_json(snapshot: SystemSnapshot, console: Optional[Console] = None) -> None:
    target = console or Console()
    target.print_json(json.dumps(snapshot.to_dict()))


def render_dashboard(
    snapshot: SystemSnapshot,
    training_statuses: Iterable[TrainingStatus] = (),
    max_command_width: int = 70,
    include_training: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    ascii_only: bool = False,
    theme: str = "soft-dark",
    display_mode: str = "auto",
    explain_layout: bool = False,
) -> Group:
    status_list = list(training_statuses)
    dashboard_statuses = _dashboard_training_statuses(status_list)
    decision = _layout_decision(width, height, display_mode)
    mode = decision.mode
    glyphs = ASCII_GLYPHS if ascii_only else UNICODE_GLYPHS
    render_theme = _resolve_theme(theme)
    if mode == "micro":
        out = _micro_dashboard(snapshot, dashboard_statuses, include_training, width, height, glyphs, render_theme)
        return _with_layout_explain(out, decision, width, render_theme, explain_layout)
    if mode == "wide-short":
        out = _wide_short_dashboard(snapshot, dashboard_statuses, include_training, width, height, glyphs, render_theme)
        return _with_layout_explain(out, decision, width, render_theme, explain_layout)
    if mode == "compact":
        out = _compact_dashboard(snapshot, dashboard_statuses, include_training, width, height, glyphs, render_theme)
        return _with_layout_explain(out, decision, width, render_theme, explain_layout)
    if mode == "medium":
        out = _medium_dashboard(snapshot, dashboard_statuses, include_training, width, height, glyphs, render_theme)
        return _with_layout_explain(out, decision, width, render_theme, explain_layout)

    training_by_pid = {status.pid: status for status in dashboard_statuses if status.pid is not None}
    process_rows = _row_budget(height, reserved=12 if include_training else 8, fallback=12)
    training_rows = _row_budget(height, reserved=14, fallback=8)
    renderables = [
        _host_panel(snapshot, render_theme),
    ]
    if include_training:
        renderables.append(_training_table(dashboard_statuses, max_rows=training_rows, theme=render_theme))
    renderables.extend(
        [
            _gpu_table(snapshot, statuses=dashboard_statuses, max_rows=_row_budget(height, reserved=10, fallback=64), theme=render_theme),
            _process_table(
                snapshot,
                training_by_pid,
                max_command_width=max_command_width,
                max_rows=process_rows,
                theme=render_theme,
            ),
        ]
    )
    renderables.append(_error_panel(snapshot, render_theme))
    out = _screen_group(renderables, width, height, render_theme)
    return _with_layout_explain(out, decision, width, render_theme, explain_layout)


def print_once(
    snapshot: SystemSnapshot,
    training_statuses: Iterable[TrainingStatus] = (),
    console: Optional[Console] = None,
    include_training: bool = True,
    ascii_only: Optional[bool] = None,
    theme: str = "soft-dark",
    display_mode: str = "auto",
    explain_layout: bool = False,
) -> None:
    target = console or Console()
    use_ascii = _console_ascii(target) if ascii_only is None else ascii_only
    target.print(
        render_dashboard(
            snapshot,
            training_statuses,
            max_command_width=max(32, target.width - 80),
            include_training=include_training,
            width=target.width,
            height=target.height,
            ascii_only=use_ascii,
            theme=theme,
            display_mode=display_mode,
            explain_layout=explain_layout,
        )
    )


def watch_plain(
    interval: float,
    backend: str,
    project_roots: Iterable[str] = (),
    show_training: bool = True,
    ascii_only: Optional[bool] = None,
    theme: str = "soft-dark",
    display_mode: str = "auto",
    explain_layout: bool = False,
    view_options: Optional[ViewOptions] = None,
    stale_after: float = 900.0,
) -> None:
    console = Console()
    use_ascii = _console_ascii(console) if ascii_only is None else ascii_only
    with Live(console=console, refresh_per_second=max(1, int(1.0 / max(interval, 0.1))), screen=True) as live:
        while True:
            snapshot = sample_once(backend=backend)
            statuses = (
                training_snapshot(project_roots=list(project_roots), processes=list(snapshot.all_processes()), stale_after=stale_after)
                if show_training
                else []
            )
            if view_options is not None:
                snapshot, statuses = apply_view(snapshot, statuses, view_options)
            live.update(
                render_dashboard(
                    snapshot,
                    statuses,
                    max_command_width=max(24, console.width - 84),
                    include_training=show_training,
                    width=console.width,
                    height=console.height,
                    ascii_only=use_ascii,
                    theme=theme,
                    display_mode=display_mode,
                    explain_layout=explain_layout,
                )
            )
            time.sleep(interval)


def _console_ascii(console: Console) -> bool:
    return bool(getattr(console.options, "ascii_only", False) or getattr(console, "legacy_windows", False))


def _layout_decision(width: Optional[int], height: Optional[int], requested: str = "auto") -> LayoutDecision:
    valid = {"auto", "micro", "wide-short", "compact", "medium", "full"}
    requested = requested if requested in valid else "auto"
    if requested != "auto":
        return LayoutDecision(requested, "forced", "requested by --mode; output may crop on small terminals", width, height)
    if height is not None and width is not None and height <= 10 and width >= 132:
        return LayoutDecision("wide-short", "auto", "height<=10 and width>=132", width, height)
    if height is not None and height <= 12:
        return LayoutDecision("micro", "auto", "height<=12", width, height)
    if width is not None and width < 72:
        return LayoutDecision("micro", "auto", "width<72", width, height)
    if (height is not None and height <= 20) or (width is not None and width < 104):
        return LayoutDecision("compact", "auto", "height<=20 or width<104", width, height)
    if (height is not None and height <= 30) or (width is not None and width < 132):
        return LayoutDecision("medium", "auto", "height<=30 or width<132", width, height)
    return LayoutDecision("full", "auto", "default full layout", width, height)


def _with_layout_explain(
    renderable: object,
    decision: LayoutDecision,
    width: Optional[int],
    theme: RenderTheme,
    enabled: bool,
) -> Group:
    if not enabled:
        return renderable if isinstance(renderable, Group) else Group(renderable)
    return Group(_layout_explain_line(decision, width, theme), renderable)


def _layout_explain_line(decision: LayoutDecision, width: Optional[int], theme: RenderTheme) -> Text:
    text = (
        f"layout {decision.mode} source={decision.source} "
        f"size={decision.width or '?'}x{decision.height or '?'} reason={decision.reason}"
    )
    return _styled_line(clamp_text(text, max(20, (width or 80) - 1)), width, theme, "muted")


def _row_budget(height: Optional[int], reserved: int, fallback: int) -> int:
    if height is None:
        return fallback
    return max(1, min(fallback, height - reserved))


def _card_row_budget(height: Optional[int]) -> int:
    if height is None:
        return 2
    return max(1, (max(1, height) - 1) // TILE_SPEC.height)


def _tile_layout(gpu_count: int, width: Optional[int]) -> tuple[int, int]:
    terminal_width = max(TILE_SPEC.min_width + 2, width or 80)
    per_tile = TILE_SPEC.target_width + TILE_SPEC.gap_width
    columns = max(1, terminal_width // per_tile)
    columns = min(max(1, gpu_count or 1), columns)
    while columns > 1:
        gap_budget = (columns - 1) * TILE_SPEC.gap_width
        if (terminal_width - gap_budget) // columns >= TILE_SPEC.min_width:
            break
        columns -= 1
    gap_budget = (columns - 1) * TILE_SPEC.gap_width
    tile_width = (terminal_width - gap_budget) // columns
    tile_width = max(TILE_SPEC.min_width, min(TILE_SPEC.max_width, tile_width))
    return columns, tile_width


def _compact_card_line_count(snapshot: SystemSnapshot, max_rows: int) -> int:
    gpus = list(snapshot.gpus)
    if not gpus:
        return 1
    visible = min(len(gpus), max_rows)
    overflow = 1 if len(gpus) > max_rows else 0
    return visible * 2 + overflow


def _screen_group(
    renderables: List[object],
    width: Optional[int],
    height: Optional[int],
    theme: RenderTheme,
    used_lines: Optional[int] = None,
) -> Group:
    items = list(renderables)
    if height is not None and used_lines is not None:
        pad_count = max(0, height - used_lines)
        items.extend(_blank_line(width, theme) for _ in range(pad_count))
    return Group(*items)


def _blank_line(width: Optional[int], theme: RenderTheme) -> Text:
    return Text(" " * max(1, width or 80), style=theme.bg_style)


def _pad_line(line: Text, width: Optional[int], theme: RenderTheme, bg: Optional[str] = None) -> Text:
    target_width = max(1, width or cell_len(line.plain) or 80)
    if cell_len(line.plain) < target_width:
        line.append(" " * (target_width - cell_len(line.plain)), style=_style(theme.text, bg or theme.background))
    return line


def _resolve_theme(name: str) -> RenderTheme:
    return THEMES.get(name, THEMES["soft-dark"])


def _style(fg: str = "", bg: str = "", extra: str = "") -> str:
    parts = []
    if extra:
        parts.append(extra)
    if fg:
        parts.append(fg)
    if bg:
        parts.append(f"on {bg}")
    return " ".join(parts)


def _role_style(theme: RenderTheme, role: str, bg: Optional[str] = None, bold: bool = False) -> str:
    colors = {
        "header": theme.text,
        "text": theme.text,
        "muted": theme.muted,
        "warn": theme.warn,
        "crit": theme.crit,
        "ok": theme.ok,
        "info": theme.info,
    }
    extra = "bold" if bold or role == "header" else ""
    return _style(colors.get(role, theme.text), theme.background if bg is None else bg, extra)


def _styled_line(
    text: str,
    width: Optional[int],
    theme: RenderTheme,
    role: str = "text",
    bg: Optional[str] = None,
    bold: bool = False,
) -> Text:
    line_width = max(1, width or cell_len(text) or 80)
    content = clamp_cells(str(text), line_width)
    content += " " * max(0, line_width - cell_len(content))
    return Text(content, style=_role_style(theme, role, bg=bg, bold=bold))


def _micro_dashboard(
    snapshot: SystemSnapshot,
    statuses: List[TrainingStatus],
    include_training: bool,
    width: Optional[int],
    height: Optional[int],
    glyphs: RenderGlyphs,
    theme: RenderTheme,
) -> Group:
    card_rows = _card_row_budget(height)
    columns, _tile_width = _tile_layout(len(snapshot.gpus), width)
    visible_tiles = min(len(snapshot.gpus), max(1, columns * max(1, card_rows)))
    rendered_card_rows = 1 if not snapshot.gpus else max(1, math.ceil(visible_tiles / columns))
    used_lines = 1 + rendered_card_rows * TILE_SPEC.height
    rows = [_host_line(snapshot, width, theme)]
    rows.append(
        _micro_gpu_blocks(
            snapshot,
            statuses if include_training else [],
            width,
            max_rows=card_rows,
            glyphs=glyphs,
            theme=theme,
        )
    )
    if include_training and statuses and height is not None and height >= 10:
        rows.append(_micro_training_table(statuses, max_rows=1, theme=theme))
        used_lines += 3
    if snapshot.errors and len(rows) < max(2, height or 99):
        rows.append(_styled_line(clamp_text(" | ".join(snapshot.errors), max(20, (width or 80) - 1)), width, theme, "warn"))
        used_lines += 1
    return _screen_group(rows, width, height, theme, used_lines=used_lines)


def _wide_short_dashboard(
    snapshot: SystemSnapshot,
    statuses: List[TrainingStatus],
    include_training: bool,
    width: Optional[int],
    height: Optional[int],
    glyphs: RenderGlyphs,
    theme: RenderTheme,
) -> Group:
    terminal_width = max(40, width or 132)
    max_rows = max(1, (height or 8) - 1)
    rows = [_host_line(snapshot, width, theme)]
    ribbons = _wide_short_ribbons(snapshot, statuses if include_training else [], terminal_width, glyphs, theme)
    hidden = max(0, len(ribbons) - max_rows)
    visible = ribbons[:max_rows]
    if hidden and visible:
        visible[-1] = _styled_line(f"+{hidden + 1} rows hidden by terminal size", terminal_width, theme, "muted")
    rows.extend(visible)
    if not ribbons:
        rows.append(_styled_line("No GPU data", terminal_width, theme, "muted"))
    return _screen_group(rows, width, height, theme, used_lines=1 + len(visible))


def _wide_short_ribbons(
    snapshot: SystemSnapshot,
    statuses: List[TrainingStatus],
    width: int,
    glyphs: RenderGlyphs,
    theme: RenderTheme,
) -> List[Text]:
    rows = []
    statuses_by_gpu = _statuses_by_gpu(statuses)
    seen_status_ids = set()
    for gpu in snapshot.gpus:
        gpu_statuses = statuses_by_gpu.get(gpu.index, [])
        for status in _active_training_statuses(gpu_statuses):
            seen_status_ids.add(id(status))
        rows.append(_wide_short_gpu_line(gpu, gpu_statuses, width, glyphs, theme))
    for status in _active_training_statuses(statuses):
        if id(status) in seen_status_ids:
            continue
        rows.append(_wide_short_status_line(status, width, theme))
    return rows


def _wide_short_gpu_line(
    gpu,
    statuses: List[TrainingStatus],
    width: int,
    glyphs: RenderGlyphs,
    theme: RenderTheme,
) -> Text:
    active_statuses = _active_training_statuses(statuses)
    status = active_statuses[0] if active_statuses else None
    health = _gpu_health(gpu, _health_training_statuses(statuses))
    parts = [
        f"G{gpu.index}",
        _health_badge(health),
        f"U{percent(gpu.utilization_gpu_percent)}",
        f"V{mb(gpu.memory_used_mb)}/{mb(gpu.memory_total_mb)}",
        f"T{na(gpu.temperature_c, 'C')}",
        f"P{_pid_label(gpu.processes)}",
    ]
    if status is not None:
        parts.extend(_run_metric_parts(status))
    else:
        parts.append("-")
    return _styled_line(_join_fit(parts, width, theme), width, theme, health.style_key, bg=theme.surface)


def _wide_short_status_line(status: TrainingStatus, width: int, theme: RenderTheme) -> Text:
    parts = [
        f"G{status.gpu_index if status.gpu_index is not None else '-'}",
        status.state.upper(),
        f"P{status.pid if status.pid is not None else '-'}",
    ]
    parts.extend(_run_metric_parts(status))
    role = "warn" if status.state in ("stalled", "orphaned") else ("crit" if status.state == "failed" else "info")
    return _styled_line(_join_fit(parts, width, theme), width, theme, role, bg=theme.surface_alt)


def _run_metric_parts(status: TrainingStatus) -> List[str]:
    parts = [
        _run_label(status.run_name, 24),
        f"E{status.epoch_label()}",
        percent(status.process_progress_percent),
    ]
    if status.eta_seconds is not None:
        parts.append(f"ETA{seconds(status.eta_seconds)}")
    if status.loss is not None:
        parts.append(f"loss{status.loss:.4g}")
    if status.learning_rate is not None:
        parts.append(f"lr{status.learning_rate:.3g}")
    if status.age_seconds is not None or status.stale_seconds is not None:
        parts.append(f"HB{seconds(status.age_seconds if status.age_seconds is not None else status.stale_seconds)}")
    if status.rank is not None:
        rank = f"r{status.rank}"
        if status.world_size is not None:
            rank += f"/{status.world_size}"
        parts.append(rank)
    return parts


def _statuses_by_gpu(statuses: List[TrainingStatus]) -> dict:
    out = {}
    for status in statuses:
        if status.gpu_index is None:
            continue
        out.setdefault(status.gpu_index, []).append(status)
    return out


def _compact_dashboard(
    snapshot: SystemSnapshot,
    statuses: List[TrainingStatus],
    include_training: bool,
    width: Optional[int],
    height: Optional[int],
    glyphs: RenderGlyphs,
    theme: RenderTheme,
) -> Group:
    max_gpu_rows = _row_budget(height, reserved=7 if include_training else 4, fallback=16)
    max_training_rows = _row_budget(height, reserved=8, fallback=8)
    gpu_lines = min(len(snapshot.gpus) or 1, max_gpu_rows) + 1
    used_lines = 1 + gpu_lines
    renderables = [
        _host_line(snapshot, width, theme),
        _compact_gpu_table(snapshot, statuses, max_rows=max_gpu_rows, glyphs=glyphs, theme=theme),
    ]
    if include_training and statuses and (height is None or height >= 16):
        renderables.append(_micro_training_table(statuses, max_rows=max_training_rows, theme=theme))
        used_lines += min(len(statuses), max_training_rows) + 2
    if snapshot.errors:
        renderables.append(_styled_line(clamp_text(" | ".join(snapshot.errors), max(20, (width or 80) - 1)), width, theme, "warn"))
        used_lines += 1
    return _screen_group(renderables, width, height, theme, used_lines=used_lines)


def _medium_dashboard(
    snapshot: SystemSnapshot,
    statuses: List[TrainingStatus],
    include_training: bool,
    width: Optional[int],
    height: Optional[int],
    glyphs: RenderGlyphs,
    theme: RenderTheme,
) -> Group:
    training_by_pid = {status.pid: status for status in statuses if status.pid is not None}
    max_gpu_rows = _row_budget(height, reserved=8, fallback=16)
    renderables = [_host_line(snapshot, width, theme)]
    if include_training and statuses and (height is None or height >= 24):
        renderables.append(_micro_training_table(statuses, max_rows=_row_budget(height, reserved=18, fallback=4), theme=theme))
    renderables.extend(
        [
            _compact_gpu_table(snapshot, statuses, max_rows=max_gpu_rows, glyphs=glyphs, theme=theme),
            _process_table(
                snapshot,
                training_by_pid,
                max_command_width=max(20, (width or 100) - 84),
                max_rows=_row_budget(height, reserved=12 if include_training else 9, fallback=8),
                theme=theme,
            ),
        ]
    )
    if snapshot.errors:
        renderables.append(_styled_line(clamp_text(" | ".join(snapshot.errors), max(20, (width or 80) - 1)), width, theme, "warn"))
    return _screen_group(renderables, width, height, theme)


def _host_line(snapshot: SystemSnapshot, width: Optional[int], theme: RenderTheme) -> Text:
    host = snapshot.host
    load = ",".join(f"{item:.1f}" for item in host.load_avg[:3]) if host.load_avg else "N/A"
    text = (
        f"gpuwatch {host.hostname} {snapshot.backend} "
        f"CPU {percent(host.cpu_percent)} RAM {percent(host.memory_percent)} "
        f"load {load}"
    )
    return _styled_line(clamp_text(text, max(20, (width or 80) - 1)), width, theme, "header")


def _micro_gpu_blocks(
    snapshot: SystemSnapshot,
    statuses: List[TrainingStatus],
    width: Optional[int],
    max_rows: int,
    glyphs: RenderGlyphs,
    theme: RenderTheme,
) -> Group:
    gpus = list(snapshot.gpus)
    columns, tile_width = _tile_layout(len(gpus), width)

    max_tiles = max(1, columns * max(1, max_rows))
    visible, overflow_count = _select_micro_gpus(gpus, statuses, max_tiles)
    tiles = [_gpu_block_lines(gpu, statuses, tile_width, glyphs, theme) for gpu in visible]
    if overflow_count:
        tiles.append(_overflow_block_lines(overflow_count, tile_width, glyphs, theme))

    rows: List[Text] = []
    target_width = max(1, width or 80)
    for start in range(0, len(tiles), columns):
        tile_row = tiles[start : start + columns]
        for line_index in range(TILE_SPEC.height):
            line = Text(style=theme.bg_style)
            for column_index in range(columns):
                if column_index:
                    line.append(" " * TILE_SPEC.gap_width, style=theme.bg_style)
                if column_index < len(tile_row):
                    line.append_text(tile_row[column_index][line_index])
                else:
                    line.append(" " * tile_width, style=theme.bg_style)
            rows.append(_pad_line(line, target_width, theme))
    if not gpus:
        rows.append(_styled_line("No GPU data", target_width, theme, "muted"))
    return Group(*rows)


def _select_micro_gpus(gpus: List[object], statuses: List[TrainingStatus], max_tiles: int) -> tuple[List[object], int]:
    if len(gpus) <= max_tiles:
        return gpus, 0
    if max_tiles <= 1:
        return sorted(gpus, key=lambda gpu: (-_gpu_interest(gpu, statuses), gpu.index))[:1], 0
    visible_slots = max_tiles - 1
    selected = sorted(gpus, key=lambda gpu: (-_gpu_interest(gpu, statuses), gpu.index))[:visible_slots]
    return sorted(selected, key=lambda gpu: gpu.index), len(gpus) - len(selected)


def _gpu_interest(gpu, statuses: List[TrainingStatus]) -> float:
    gpu_statuses = _health_training_statuses([status for status in statuses if status.gpu_index == gpu.index])
    score = float(_gpu_health(gpu, gpu_statuses).priority)
    if gpu_statuses:
        score += 100.0
    if gpu.processes:
        score += 50.0
    if gpu.utilization_gpu_percent is not None:
        score += min(10.0, float(gpu.utilization_gpu_percent) / 10.0)
    if gpu.memory_percent is not None:
        score += min(10.0, float(gpu.memory_percent) / 10.0)
    return score


def _gpu_block(gpu, statuses: List[TrainingStatus], tile_width: int, glyphs: RenderGlyphs, theme: RenderTheme) -> Text:
    return Text.assemble(*_interleave_lines(_gpu_block_lines(gpu, statuses, tile_width, glyphs, theme)))


def _gpu_block_lines(gpu, statuses: List[TrainingStatus], tile_width: int, glyphs: RenderGlyphs, theme: RenderTheme) -> List[Text]:
    all_gpu_statuses = [status for status in statuses if status.gpu_index == gpu.index]
    health = _gpu_health(gpu, _health_training_statuses(all_gpu_statuses))
    train = _tile_training_label(all_gpu_statuses)
    pid = _pid_label(gpu.processes)
    used = mb(gpu.memory_used_mb)
    total = mb(gpu.memory_total_mb)
    line1 = _card_top(f"G{gpu.index} · {_health_badge(health)}", tile_width, glyphs, theme, health)
    line2 = _card_body(
        _join_fit(
            [
                f"U {percent(gpu.utilization_gpu_percent)}",
                f"V {used}/{total}",
            ],
            tile_width - 2,
            theme,
        ),
        tile_width,
        glyphs,
        theme,
        alt=False,
    )
    line3_content = _join_fit(
        [
            f"T {na(gpu.temperature_c, 'C')}",
            f"P {pid}",
            f"E {train[1:] if train.startswith('E') else (train or '-')}",
        ],
        tile_width - 4,
        theme,
    )
    line3 = _card_bottom(line3_content, tile_width, glyphs, theme)
    return [line1, line2, line3]


def _overflow_block(hidden: int, tile_width: int, glyphs: RenderGlyphs, theme: RenderTheme) -> Text:
    return Text.assemble(*_interleave_lines(_overflow_block_lines(hidden, tile_width, glyphs, theme)))


def _overflow_block_lines(hidden: int, tile_width: int, glyphs: RenderGlyphs, theme: RenderTheme) -> List[Text]:
    health = GpuHealth("MORE", 0, "muted")
    line1 = _card_top(f"+{hidden} GPUs", tile_width, glyphs, theme, health)
    line2 = _card_body("hidden by terminal size", tile_width, glyphs, theme, alt=False)
    line3 = _card_bottom("", tile_width, glyphs, theme)
    return [line1, line2, line3]


def _interleave_lines(lines: List[Text]) -> List[object]:
    out: List[object] = []
    for index, line in enumerate(lines):
        if index:
            out.append("\n")
        out.append(line)
    return out


def _gpu_health(gpu, statuses: List[TrainingStatus]) -> GpuHealth:
    states = {status.state.lower() for status in statuses if status.state}
    temp = gpu.temperature_c
    util = gpu.utilization_gpu_percent
    memory = gpu.memory_percent
    if "failed" in states:
        return GpuHealth("CRIT", 5, "crit", "failed")
    if temp is not None and temp >= 82:
        return GpuHealth("CRIT", 5, "crit", f"{temp}C")
    if memory is not None and memory >= 97:
        return GpuHealth("CRIT", 5, "crit", f"mem{memory:.0f}")
    if "stalled" in states:
        age = max((status.age_seconds or status.stale_seconds or 0 for status in statuses), default=0)
        return GpuHealth("WARN", 4, "warn", f"stale{seconds(age)}")
    if memory is not None and memory >= 90:
        return GpuHealth("WARN", 4, "warn", f"mem{memory:.0f}")
    if temp is not None and temp >= 72:
        return GpuHealth("HOT", 3, "warn", f"{temp}C")
    if util is not None and util >= 50:
        return GpuHealth("BUSY", 2, "info", f"u{util:.0f}")
    if (util is None or util < 5) and (memory is None or memory < 5) and not gpu.processes:
        return GpuHealth("IDLE", 1, "muted", "idle")
    return GpuHealth("OK", 0, "ok", "")


def _health_badge(health: GpuHealth) -> str:
    return f"{health.label} {health.reason}" if health.reason else health.label


def _card_top(title: str, width: int, glyphs: RenderGlyphs, theme: RenderTheme, health: GpuHealth) -> Text:
    inner_width = max(0, width - 2)
    label = f" {title} "
    label = clamp_cells(label, inner_width, marker=glyphs.marker)
    fill = glyphs.horizontal * max(0, inner_width - cell_len(label))
    line = Text(style=theme.surface_style)
    line.append(glyphs.top_left, style=theme.border_style)
    line.append(label, style=_role_style(theme, health.style_key, bg=theme.surface, bold=True))
    line.append(fill, style=theme.border_style)
    line.append(glyphs.top_right, style=theme.border_style)
    return line


def _card_body(content: str, width: int, glyphs: RenderGlyphs, theme: RenderTheme, alt: bool = False) -> Text:
    inner_width = max(0, width - 2)
    bg = theme.surface_alt if alt else theme.surface
    body = clamp_cells(content, inner_width, marker=glyphs.marker)
    body += " " * max(0, inner_width - cell_len(body))
    line = Text(style=_style(theme.text, bg))
    line.append(glyphs.vertical, style=_style(theme.border, bg))
    line.append(body, style=_style(theme.text, bg))
    line.append(glyphs.vertical, style=_style(theme.border, bg))
    return line


def _card_bottom(content: str, width: int, glyphs: RenderGlyphs, theme: RenderTheme) -> Text:
    inner_width = max(0, width - 2)
    label = f" {content} " if content else ""
    label = clamp_cells(label, inner_width, marker=glyphs.marker)
    fill = glyphs.horizontal * max(0, inner_width - cell_len(label))
    line = Text(style=theme.surface_style)
    line.append(glyphs.bottom_left, style=theme.border_style)
    if label:
        line.append(label, style=_style(theme.muted, theme.surface))
    line.append(fill, style=theme.border_style)
    line.append(glyphs.bottom_right, style=theme.border_style)
    return line


def _join_fit(parts: List[str], width: int, theme: RenderTheme) -> str:
    del theme
    out = ""
    for part in parts:
        if not part:
            continue
        candidate = part if not out else f"{out}  {part}"
        if cell_len(candidate) <= width:
            out = candidate
    return clamp_cells(out or "-", max(1, width))


def _compact_gpu_cards(
    snapshot: SystemSnapshot,
    statuses: List[TrainingStatus],
    max_rows: int,
    glyphs: RenderGlyphs,
    theme: RenderTheme,
    width: Optional[int],
) -> Table:
    terminal_width = max(28, width or 80)
    card_width = min(max(terminal_width, TILE_SPEC.min_width), terminal_width)
    table = Table.grid(padding=(0, 0))
    table.style = theme.bg_style
    table.add_column(width=card_width, no_wrap=True, overflow="crop")
    gpus = list(snapshot.gpus)
    visible = gpus[:max_rows]
    for gpu in visible:
        table.add_row(_compact_gpu_card(gpu, statuses, card_width, glyphs, theme))
    if len(gpus) > max_rows:
        table.add_row(_styled_line(f"+{len(gpus) - max_rows} GPUs hidden by terminal size", card_width, theme, "muted"))
    if not gpus:
        table.add_row(_styled_line("No GPU data", card_width, theme, "muted"))
    return table


def _compact_gpu_card(gpu, statuses: List[TrainingStatus], width: int, glyphs: RenderGlyphs, theme: RenderTheme) -> Text:
    gpu_statuses = [status for status in statuses if status.gpu_index == gpu.index]
    health = _gpu_health(gpu, _health_training_statuses(gpu_statuses))
    pid = _pid_label(gpu.processes)
    training = _gpu_training_label(gpu.index, statuses)
    name_width = max(8, width // 4)
    line1 = _compact_body_line(
        _join_fit(
            [
                f"G{gpu.index} {_health_badge(health)}",
                short_gpu_name(gpu.name),
                f"U {percent(gpu.utilization_gpu_percent)}",
                f"V {mb(gpu.memory_used_mb)}/{mb(gpu.memory_total_mb)}",
                f"T {na(gpu.temperature_c, 'C')}",
            ],
            width,
            theme,
        ),
        width,
        glyphs,
        theme,
        health.style_key,
        bg=theme.surface,
        bold=True,
    )
    line2 = _compact_body_line(
        _join_fit(
            [
                clamp_cells(gpu.name, name_width),
                f"Fan {percent(gpu.fan_percent)}",
                f"Power {_power(gpu.power_draw_w, gpu.power_limit_w)}",
                f"PIDs {pid}",
                f"Train {training}",
            ],
            width,
            theme,
        ),
        width,
        glyphs,
        theme,
        "muted",
        bg=theme.surface_alt,
    )
    return Text.assemble(line1, "\n", line2)


def _compact_body_line(
    content: str,
    width: int,
    glyphs: RenderGlyphs,
    theme: RenderTheme,
    role: str,
    bg: str,
    bold: bool = False,
) -> Text:
    inner_width = max(0, width - 2)
    body = clamp_cells(content, inner_width, marker=glyphs.marker)
    body += " " * max(0, inner_width - cell_len(body))
    line = Text(style=_style(theme.text, bg))
    line.append(glyphs.vertical, style=_style(theme.border, bg))
    line.append(body, style=_role_style(theme, role, bg=bg, bold=bold))
    line.append(glyphs.vertical, style=_style(theme.border, bg))
    return line


def _compact_gpu_table(
    snapshot: SystemSnapshot,
    statuses: List[TrainingStatus],
    max_rows: int,
    glyphs: RenderGlyphs,
    theme: Optional[RenderTheme] = None,
) -> Table:
    theme = theme or _resolve_theme("soft-dark")
    table = Table(
        title=None,
        expand=True,
        box=None,
        pad_edge=False,
        style=theme.surface_style,
        header_style=_style(theme.text, theme.surface_alt, "bold"),
        row_styles=[theme.surface_style],
    )
    table.add_column("GPU", no_wrap=True, justify="right")
    table.add_column("State", no_wrap=True)
    table.add_column("Util", no_wrap=True)
    table.add_column("VRAM", no_wrap=True)
    table.add_column("Temp", no_wrap=True)
    table.add_column("PIDs", no_wrap=True)
    table.add_column("Run", no_wrap=True)
    table.add_column("Epoch", no_wrap=True)
    table.add_column("Prog", no_wrap=True)
    table.add_column("ETA", no_wrap=True)
    gpus = list(snapshot.gpus)
    for gpu in gpus[:max_rows]:
        all_gpu_statuses = [status for status in statuses if status.gpu_index == gpu.index]
        gpu_statuses = _active_training_statuses(all_gpu_statuses)
        status = gpu_statuses[0] if gpu_statuses else None
        health = _gpu_health(gpu, _health_training_statuses(all_gpu_statuses))
        table.add_row(
            _gpu_badge(gpu.index, glyphs),
            _health_badge(health),
            Text(f"{bar(gpu.utilization_gpu_percent, 8)} {percent(gpu.utilization_gpu_percent)}"),
            f"{mb(gpu.memory_used_mb)}/{mb(gpu.memory_total_mb)}",
            na(gpu.temperature_c, "C"),
            _pid_label(gpu.processes),
            _run_label(status.run_name if status else None, 18),
            status.epoch_label() if status else "-",
            percent(status.process_progress_percent if status else None) if status else "-",
            seconds(status.eta_seconds) if status and status.eta_seconds is not None else "-",
        )
    if len(gpus) > max_rows:
        table.add_row("...", f"+{len(gpus) - max_rows} more", "", "", "", "", "", "", "", "")
    if not gpus:
        table.add_row("-", "No GPU data", "", "", "", "", "", "", "", "")
    return table


def _micro_training_table(statuses: List[TrainingStatus], max_rows: int, theme: RenderTheme) -> Table:
    table = Table(
        title=None,
        expand=True,
        box=None,
        pad_edge=False,
        style=theme.surface_style,
        header_style=_style(theme.text, theme.surface_alt, "bold"),
        row_styles=[theme.surface_style],
    )
    table.add_column("GPU", no_wrap=True, justify="right")
    table.add_column("PID", no_wrap=True, justify="right")
    table.add_column("Run", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Epoch", no_wrap=True)
    table.add_column("Progress", no_wrap=True)
    table.add_column("ETA", no_wrap=True)
    table.add_column("HB", no_wrap=True)
    table.add_column("Reason", no_wrap=True)
    for status in statuses[:max_rows]:
        table.add_row(
            str(status.gpu_index) if status.gpu_index is not None else "-",
            str(status.pid) if status.pid is not None else "-",
            _run_label(status.run_name, 24),
            status.state,
            status.epoch_label(),
            percent(status.process_progress_percent),
            seconds(status.eta_seconds) if status.eta_seconds is not None else "-",
            seconds(status.age_seconds if status.age_seconds is not None else status.stale_seconds),
            status.state_reason or status.phase,
        )
    if len(statuses) > max_rows:
        table.add_row("...", f"+{len(statuses) - max_rows}", "", "", "", "", "", "", "")
    return table


def _host_panel(snapshot: SystemSnapshot, theme: RenderTheme) -> Panel:
    host = snapshot.host
    load = " ".join(f"{item:.2f}" for item in host.load_avg) if host.load_avg else "N/A"
    text = (
        f"{host.hostname} | backend={snapshot.backend} | "
        f"CPU {percent(host.cpu_percent)} | RAM {percent(host.memory_percent)} "
        f"({mb(host.memory_used_mb)}/{mb(host.memory_total_mb)}) | load {load}"
    )
    return Panel(
        Text(text, style=theme.surface_style),
        title="gpuwatch",
        padding=(0, 1),
        style=theme.surface_style,
        border_style=theme.border_style,
    )


def _gpu_table(
    snapshot: SystemSnapshot,
    statuses: Optional[List[TrainingStatus]] = None,
    max_rows: int = 64,
    theme: Optional[RenderTheme] = None,
) -> Table:
    theme = theme or _resolve_theme("soft-dark")
    table = Table(
        title="GPUs",
        expand=True,
        style=theme.surface_style,
        header_style=_style(theme.text, theme.surface_alt, "bold"),
        border_style=theme.border_style,
        row_styles=[theme.surface_style],
    )
    table.add_column("GPU", no_wrap=True, justify="right")
    table.add_column("State", no_wrap=True)
    table.add_column("Name")
    table.add_column("Util", no_wrap=True)
    table.add_column("VRAM", no_wrap=True)
    table.add_column("Mem", no_wrap=True)
    table.add_column("Fan", no_wrap=True)
    table.add_column("Temp", no_wrap=True)
    table.add_column("Power", no_wrap=True)
    table.add_column("PIDs")
    gpus = list(snapshot.gpus)
    statuses_by_gpu = _statuses_by_gpu(statuses or [])
    for gpu in gpus[:max_rows]:
        health = _gpu_health(gpu, statuses_by_gpu.get(gpu.index, []))
        pids = ", ".join(str(process.pid) for process in gpu.processes) or "-"
        memory_label = f"{mb(gpu.memory_used_mb)}/{mb(gpu.memory_total_mb)}"
        table.add_row(
            str(gpu.index),
            _health_badge(health),
            gpu.name,
            Text(f"{bar(gpu.utilization_gpu_percent, 12)} {percent(gpu.utilization_gpu_percent)}"),
            Text(f"{bar(gpu.memory_percent, 12)} {percent(gpu.memory_percent)}"),
            memory_label,
            percent(gpu.fan_percent),
            na(gpu.temperature_c, "C"),
            _power(gpu.power_draw_w, gpu.power_limit_w),
            pids,
        )
    if len(gpus) > max_rows:
        table.add_row("...", f"+{len(gpus) - max_rows} more", "", "", "", "", "", "", "", "")
    if not gpus:
        table.add_row("-", "-", "No GPU data", "-", "-", "-", "-", "-", "-", "-")
    return table


def _process_table(
    snapshot: SystemSnapshot,
    training_by_pid: dict,
    max_command_width: int,
    max_rows: int = 12,
    theme: Optional[RenderTheme] = None,
) -> Table:
    theme = theme or _resolve_theme("soft-dark")
    table = Table(
        title="GPU Processes",
        expand=True,
        style=theme.surface_style,
        header_style=_style(theme.text, theme.surface_alt, "bold"),
        border_style=theme.border_style,
        row_styles=[theme.surface_style],
    )
    table.add_column("GPU", no_wrap=True, justify="right")
    table.add_column("PID", no_wrap=True, justify="right")
    table.add_column("User", no_wrap=True)
    table.add_column("GPU Mem", no_wrap=True, justify="right")
    table.add_column("CPU", no_wrap=True, justify="right")
    table.add_column("RSS", no_wrap=True, justify="right")
    table.add_column("Time", no_wrap=True)
    table.add_column("Training", no_wrap=True)
    table.add_column("Command")

    rows = 0
    processes = sorted(snapshot.all_processes(), key=lambda item: (item.gpu_index, item.pid))
    for process in processes[:max_rows]:
        status = training_by_pid.get(process.pid)
        training_label = status.compact_label() if status else "-"
        table.add_row(
            str(process.gpu_index),
            str(process.pid),
            process.username or "?",
            mb(process.gpu_memory_mb),
            percent(process.cpu_percent),
            mb(process.rss_mb),
            seconds(process.elapsed_s),
            training_label,
            clamp_text(process.command, max_command_width),
        )
        rows += 1
    if len(processes) > max_rows:
        table.add_row("...", f"+{len(processes) - max_rows}", "", "", "", "", "", "", "")
    if rows == 0:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "No GPU processes")
    return table


def _training_table(
    statuses: Iterable[TrainingStatus],
    max_rows: int = 8,
    theme: Optional[RenderTheme] = None,
) -> Table:
    theme = theme or _resolve_theme("soft-dark")
    status_list = list(statuses)
    table = Table(
        title="Training Progress",
        expand=True,
        style=theme.surface_style,
        header_style=_style(theme.text, theme.surface_alt, "bold"),
        border_style=theme.border_style,
        row_styles=[theme.surface_style],
    )
    table.add_column("GPU", no_wrap=True, justify="right")
    table.add_column("PID", no_wrap=True, justify="right")
    table.add_column("Run")
    table.add_column("State", no_wrap=True)
    table.add_column("Phase", no_wrap=True)
    table.add_column("Epoch", no_wrap=True)
    table.add_column("Progress", no_wrap=True)
    table.add_column("ETA", no_wrap=True)
    table.add_column("Speed", no_wrap=True)
    table.add_column("HB", no_wrap=True)
    table.add_column("Metric")
    table.add_column("Evidence")
    for status in status_list[:max_rows]:
        progress = status.process_progress_percent
        metric = status.metric_label()
        table.add_row(
            str(status.gpu_index) if status.gpu_index is not None else "-",
            str(status.pid) if status.pid is not None else "-",
            _run_label(status.run_name, 36),
            status.state,
            status.phase,
            status.epoch_label(),
            Text(f"{bar(progress, 16)} {percent(progress)}"),
            seconds(status.eta_seconds) if status.eta_seconds is not None else "-",
            _speed_label(status),
            seconds(status.age_seconds if status.age_seconds is not None else status.stale_seconds),
            metric,
            f"{status.confidence:.2f} {status.evidence_label()}",
        )
    if len(status_list) > max_rows:
        table.add_row("...", f"+{len(status_list) - max_rows}", "", "", "", "", "", "", "", "", "", "")
    if not status_list:
        table.add_row("-", "-", "No training status", "-", "-", "-", "-", "-", "-", "-", "-", "-")
    return table


def _speed_label(status: TrainingStatus) -> str:
    if status.speed_per_second is None:
        return "-"
    unit = status.speed_unit or "step"
    suffix = "it/s" if unit == "step" else f"{unit}/s"
    return f"{status.speed_per_second:.2g} {suffix}"


def _run_label(run_name: Optional[str], width: int) -> str:
    text = run_name or "-"
    if cell_len(text) <= width:
        return text
    if width <= 1:
        return "~"[:width]
    tail = text[-(width - 1) :]
    return "~" + tail


def _error_panel(snapshot: SystemSnapshot, theme: RenderTheme) -> Panel:
    if not snapshot.errors:
        return Panel(
            Text("OK", style=_style(theme.ok, theme.surface, "bold")),
            title="Status",
            padding=(0, 1),
            style=theme.surface_style,
            border_style=theme.border_style,
        )
    return Panel(
        "\n".join(snapshot.errors),
        title="Status",
        padding=(0, 1),
        style=theme.surface_style,
        border_style=_style(theme.warn, theme.surface),
    )


def _power(draw: Optional[float], limit: Optional[float]) -> str:
    if draw is None and limit is None:
        return "N/A"
    if limit is None:
        return f"{draw:.1f}W"
    if draw is None:
        return f"?/{limit:.0f}W"
    return f"{draw:.1f}/{limit:.0f}W"


def _pid_label(processes: Iterable[GpuProcessSnapshot]) -> str:
    pids = [str(process.pid) for process in processes]
    if not pids:
        return "-"
    if len(pids) <= 3:
        return ",".join(pids)
    return ",".join(pids[:3]) + f"+{len(pids) - 3}"


def _gpu_training_label(gpu_index: int, statuses: List[TrainingStatus]) -> str:
    gpu_statuses = _active_training_statuses([status for status in statuses if status.gpu_index == gpu_index])
    if not gpu_statuses:
        return "-"
    labels = [status.compact_label() for status in gpu_statuses[:2]]
    if len(gpu_statuses) > 2:
        labels.append(f"+{len(gpu_statuses) - 2}")
    return " ".join(labels)


def _tile_training_label(statuses: List[TrainingStatus]) -> str:
    statuses = _active_training_statuses(statuses)
    if not statuses:
        return ""
    status = statuses[0]
    label = status.epoch_label()
    if status.process_progress_percent is not None:
        label = f"{label}/{status.process_progress_percent:.0f}%"
    if len(statuses) > 1:
        label += f"+{len(statuses) - 1}"
    return f"E{label}"


def _active_training_statuses(statuses: List[TrainingStatus]) -> List[TrainingStatus]:
    inactive = {"complete", "failed", "orphaned"}
    return [status for status in statuses if (status.state or "").lower() not in inactive]


def _dashboard_training_statuses(statuses: List[TrainingStatus]) -> List[TrainingStatus]:
    hidden = {"complete", "orphaned"}
    return [status for status in statuses if (status.state or "").lower() not in hidden]


def _health_training_statuses(statuses: List[TrainingStatus]) -> List[TrainingStatus]:
    inactive = {"complete", "orphaned"}
    return [status for status in statuses if (status.state or "").lower() not in inactive]


def short_gpu_name(name: str) -> str:
    parts = [part for part in name.replace("NVIDIA", "").split() if part]
    if not parts:
        return "GPU"
    if len(parts) == 1:
        return parts[0]
    return "".join(part[0] for part in parts[:-1]) + parts[-1]


def _append(target: Text, value: str, style: Optional[str] = None) -> None:
    target.append(str(value), style=style)


def _append_metric_parts(target: Text, parts: List[tuple[str, str]], tile_width: int, glyphs: RenderGlyphs) -> None:
    for index, (label, style) in enumerate(parts):
        prefix = f" {glyphs.section} " if index else ""
        if cell_len(target.plain) + cell_len(prefix) + cell_len(label) > tile_width:
            continue
        if prefix:
            _append(target, prefix, "dim")
        _append(target, label, style)


def _fit_text(text: Text, width: int, marker: str) -> Text:
    plain = text.plain
    plain_width = cell_len(plain)
    if plain_width == width:
        return text
    if plain_width < width:
        out = text.copy()
        out.append(" " * (width - plain_width))
        return out
    return Text(clamp_cells(plain, width, marker=marker), style=text.style)


def _metric_style(value: Optional[float], warn: float, crit: float) -> str:
    if value is None:
        return "dim"
    if value >= crit:
        return "bold red"
    if value >= warn:
        return "yellow"
    return "cyan"


def _gpu_badge(index: int, glyphs: RenderGlyphs) -> Text:
    badge = Text()
    _append(badge, glyphs.tile_left, "dim")
    _append(badge, str(index), f"bold {GPU_BADGE_STYLES[index % len(GPU_BADGE_STYLES)]}")
    _append(badge, glyphs.tile_right, "dim")
    return badge
