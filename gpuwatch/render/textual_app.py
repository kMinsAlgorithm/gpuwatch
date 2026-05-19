from __future__ import annotations

from typing import Iterable

from gpuwatch.render.rich_cli import THEMES, render_dashboard
from gpuwatch.sampler import sample_once
from gpuwatch.training.status import snapshot as training_snapshot


def run_textual(
    interval: float,
    backend: str,
    project_roots: Iterable[str] = (),
    show_training: bool = True,
    ascii_only: bool = False,
    theme: str = "soft-dark",
) -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Container
        from textual.reactive import reactive
        from textual.widgets import Footer, Static
    except Exception as exc:
        raise RuntimeError("Textual is not installed. Install gpuwatch[tui] or use --plain.") from exc

    roots = list(project_roots)
    render_theme = THEMES.get(theme, THEMES["soft-dark"])
    css_background = render_theme.background or "#000000"
    css_surface = render_theme.surface or css_background
    css_text = render_theme.text or "#ffffff"

    class Dashboard(Static):
        snapshot_text = reactive("")

        def render(self):
            snapshot = sample_once(backend=backend)
            statuses = (
                training_snapshot(project_roots=roots, processes=list(snapshot.all_processes()))
                if show_training
                else []
            )
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
        Footer {{
            background: {css_surface};
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
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(interval, self.refresh_dashboard)

        def refresh_dashboard(self) -> None:
            if not self.paused:
                self.query_one(Dashboard).refresh()

        def action_toggle_pause(self) -> None:
            self.paused = not self.paused

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
