from __future__ import annotations

from typing import Iterable

from rich.text import Text

from gpuwatch.render.rich_cli import THEMES, render_dashboard
from gpuwatch.sampler import sample_once
from gpuwatch.training.status import snapshot as training_snapshot
from gpuwatch.view import ViewOptions, apply_view


def run_textual(
    interval: float,
    backend: str,
    project_roots: Iterable[str] = (),
    show_training: bool = True,
    ascii_only: bool = False,
    theme: str = "soft-dark",
    display_mode: str = "auto",
    explain_layout: bool = False,
    view_options: ViewOptions = None,
    fallback_to_fake: bool = False,
) -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Container
        from textual.reactive import reactive
        from textual.widgets import Static
    except Exception as exc:
        raise RuntimeError("Textual is not installed. Install gpuwatch[tui] or use --plain.") from exc

    roots = list(project_roots)
    render_theme = THEMES.get(theme, THEMES["soft-dark"])
    css_background = render_theme.background or "#000000"
    css_surface = render_theme.surface or css_background
    css_surface_alt = render_theme.surface_alt or css_surface
    css_text = render_theme.text or "#ffffff"
    css_muted = render_theme.muted or css_text
    css_warn = render_theme.warn or css_text

    class Dashboard(Static):
        snapshot_text = reactive("")

        def render(self):
            snapshot = sample_once(backend=backend, fallback_to_fake=fallback_to_fake)
            statuses = (
                training_snapshot(project_roots=roots, processes=list(snapshot.all_processes()))
                if show_training
                else []
            )
            if view_options is not None:
                snapshot, statuses = apply_view(snapshot, statuses, view_options)
            width = max(24, self.size.width - 80)
            return render_dashboard(
                snapshot,
                statuses,
                max_command_width=width,
                include_training=show_training,
                width=self.size.width,
                height=self.size.height,
                ascii_only=ascii_only or _app_ascii(self.app),
                theme=theme,
                display_mode=display_mode,
                explain_layout=explain_layout,
            )

    class GpuwatchApp(App):
        CSS = f"""
        Screen {{
            layout: vertical;
            background: {css_background};
            color: {css_text};
        }}
        #root {{
            width: 100%;
            height: 1fr;
            background: {css_background};
        }}
        Dashboard {{
            width: 100%;
            height: 1fr;
            overflow: hidden;
            background: {css_background};
            color: {css_text};
        }}
        ControlFooter {{
            dock: bottom;
            height: 1;
            width: 100%;
            background: {css_background};
            color: {css_text};
        }}
        """
        BINDINGS = [
            Binding("q", "quit", "Quit", priority=True),
            Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
            Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
            Binding("p", "toggle_pause", "Pause"),
        ]

        paused = reactive(False)

        def compose(self) -> ComposeResult:
            with Container(id="root"):
                yield Dashboard()
            yield ControlFooter()

        def on_mount(self) -> None:
            self.set_interval(interval, self.refresh_dashboard)

        def refresh_dashboard(self) -> None:
            if not self.paused:
                self.query_one(Dashboard).refresh()

        def action_toggle_pause(self) -> None:
            self.paused = not self.paused
            self.query_one(ControlFooter).refresh()

    class ControlFooter(Static):
        def render(self):
            paused = bool(getattr(self.app, "paused", False))
            line = Text(style=f"{css_text} on {css_background}")
            line.append(" q ", style=f"bold {css_warn} on {css_surface_alt}")
            line.append("Quit  ", style=f"{css_text} on {css_background}")
            line.append(" p ", style=f"bold {css_warn} on {css_surface_alt}")
            line.append("Resume" if paused else "Pause", style=f"{css_text} on {css_background}")
            if paused:
                line.append("  paused", style=f"{css_muted} on {css_background}")
            return line

    try:
        GpuwatchApp().run()
    finally:
        _restore_terminal_after_textual()


def _app_ascii(app) -> bool:
    console = getattr(app, "console", None)
    options = getattr(console, "options", None)
    return bool(getattr(options, "ascii_only", False) or getattr(console, "legacy_windows", False))


def _restore_terminal_after_textual() -> None:
    try:
        import sys

        if sys.stdout.isatty():
            sys.stdout.write("\033[0m\033[?25h\033[2J\033[H")
            sys.stdout.flush()
    except OSError:
        pass
