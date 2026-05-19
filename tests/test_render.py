import unittest
from io import StringIO
from dataclasses import replace

from rich.cells import cell_len
from rich.console import Console

from gpuwatch.render.rich_cli import THEMES, render_dashboard
from gpuwatch.sampler import sample_once
from gpuwatch.training.status import TrainingStatus


def render_text(snapshot, statuses=(), width=160, height=10, ascii_only=False, theme="soft-dark", display_mode="auto", explain_layout=False):
    console = Console(width=width, height=height, color_system=None, no_color=True, record=True, file=StringIO())
    console.print(
        render_dashboard(
            snapshot,
            statuses,
            include_training=True,
            width=width,
            height=height,
            ascii_only=ascii_only,
            theme=theme,
            display_mode=display_mode,
            explain_layout=explain_layout,
        )
    )
    return console.export_text()


def snapshot_with_gpus(count):
    snapshot = sample_once(backend="fake")
    gpus = []
    for index in range(count):
        base = snapshot.gpus[index % len(snapshot.gpus)]
        gpus.append(
            replace(
                base,
                index=index,
                uuid=f"GPU-TEST-{index}",
                name="Very Long NVIDIA Test GPU Name",
                utilization_gpu_percent=(index * 17) % 100,
                memory_used_mb=512 * index,
                temperature_c=30 + index,
                fan_percent=None if index % 3 == 0 else 35 + index,
            )
        )
    return replace(snapshot, gpus=tuple(gpus))


class RenderTests(unittest.TestCase):
    def test_micro_unicode_cards_have_visible_boundaries(self):
        text = render_text(snapshot_with_gpus(8), width=204, height=8, ascii_only=False, display_mode="micro")
        self.assertIn("┌ G0", text)
        self.assertIn("┌ G5", text)
        self.assertIn("┌ G7", text)
        self.assertIn("│U", text)
        self.assertIn("└ T", text)
        self.assertGreaterEqual(text.count("┌ G"), 8)
        first_card_line = next(line for line in text.splitlines() if line.startswith("┌ G0"))
        self.assertEqual(first_card_line.count("┌ G"), 6)
        for line in text.splitlines():
            self.assertLessEqual(cell_len(line), 204)

    def test_micro_cards_fit_common_small_viewports(self):
        snapshot = snapshot_with_gpus(8)
        for width, height in ((160, 10), (100, 8), (80, 12)):
            text = render_text(snapshot, width=width, height=height, ascii_only=False, display_mode="micro")
            self.assertIn("┌ G", text)
            self.assertIn("│U", text)
            for line in text.splitlines():
                self.assertLessEqual(cell_len(line), width)

    def test_micro_prioritizes_training_gpu_when_overflowing(self):
        snapshot = snapshot_with_gpus(8)
        statuses = [
            TrainingStatus(
                pid=12345,
                gpu_index=3,
                run_name="tiny-overflow",
                phase="train",
                epoch=4,
                max_epoch=9,
                process_progress_percent=44.0,
                state="running",
            )
        ]
        text = render_text(snapshot, statuses=statuses, width=80, height=8, ascii_only=False)
        self.assertIn("┌ G3", text)
        self.assertIn("E 4/9/44%", text)
        self.assertIn("GPUs", text)
        for line in text.splitlines():
            self.assertLessEqual(cell_len(line), 80)

    def test_micro_does_not_show_orphaned_training_as_active_epoch(self):
        snapshot = snapshot_with_gpus(4)
        statuses = [
            TrainingStatus(
                pid=99999999,
                gpu_index=3,
                run_name="gone",
                phase="train",
                epoch=4,
                max_epoch=9,
                process_progress_percent=44.0,
                state="orphaned",
            )
        ]
        text = render_text(snapshot, statuses=statuses, width=204, height=8, ascii_only=False)
        self.assertNotIn("E 4/9/44%", text)

    def test_dashboard_hides_orphaned_training_history(self):
        statuses = [
            TrainingStatus(
                pid=99999999,
                gpu_index=3,
                run_name="old_heartbeat",
                phase="train",
                epoch=4,
                max_epoch=9,
                process_progress_percent=44.0,
                eta_seconds=60,
                age_seconds=1800,
                state="orphaned",
                state_reason="pid_missing",
            )
        ]
        text = render_text(snapshot_with_gpus(4), statuses=statuses, width=160, height=28, ascii_only=False)
        self.assertNotIn("old_heartbeat", text)
        self.assertNotIn("orphaned", text)
        self.assertNotIn("pid_missing", text)

    def test_compact_handles_status_without_run_name(self):
        statuses = [
            TrainingStatus(
                pid=12345,
                gpu_index=1,
                run_name=None,
                phase="unknown",
                state="unbound",
                state_reason="gpu_process_only",
            )
        ]
        text = render_text(snapshot_with_gpus(4), statuses=statuses, width=100, height=18, display_mode="compact")
        self.assertIn("GPU", text)
        self.assertIn("unbound", text)
        for line in text.splitlines():
            self.assertLessEqual(cell_len(line), 100)

    def test_micro_ascii_tiles_avoid_unicode(self):
        text = render_text(snapshot_with_gpus(4), width=160, height=10, ascii_only=True, display_mode="micro")
        for index in range(4):
            self.assertIn(f"+ G{index}", text)
        self.assertNotIn("┃", text)
        self.assertNotIn("│", text)
        self.assertNotIn("…", text)
        self.assertNotIn("┌", text)
        self.assertIn("|U", text)

    def test_many_gpu_overflow_tile(self):
        text = render_text(snapshot_with_gpus(16), width=80, height=4, ascii_only=False)
        self.assertIn("GPUs", text)
        self.assertIn("hidden by terminal size", text)
        for line in text.splitlines():
            self.assertLessEqual(cell_len(line), 80)

    def test_full_mode_bars_render_literal_hashes(self):
        text = render_text(snapshot_with_gpus(2), width=150, height=40, ascii_only=False)
        self.assertIn("[", text)
        self.assertIn("#", text)
        self.assertIn("]", text)

    def test_edge_widths_do_not_crash_or_overflow(self):
        snapshot = snapshot_with_gpus(8)
        for width, height in ((20, 8), (40, 10), (200, 10)):
            text = render_text(snapshot, width=width, height=height, ascii_only=True)
            for line in text.splitlines():
                self.assertLessEqual(cell_len(line), width)

    def test_no_gpu_state(self):
        snapshot = replace(sample_once(backend="fake"), gpus=())
        text = render_text(snapshot, width=80, height=8, ascii_only=True)
        self.assertIn("No GPU data", text)

    def test_soft_dark_theme_has_background_styles(self):
        theme = THEMES["soft-dark"]
        self.assertEqual(theme.background, "#111827")
        self.assertEqual(theme.surface_alt, "#243244")
        self.assertIn("on #111827", theme.bg_style)
        self.assertIn("on #1f2937", theme.surface_style)
        self.assertIn("on #243244", theme.alt_style)

    def test_health_badges_cover_busy_hot_warn_crit_idle(self):
        snapshot = snapshot_with_gpus(5)
        gpus = list(snapshot.gpus)
        gpus[0] = replace(gpus[0], utilization_gpu_percent=0, memory_used_mb=0, temperature_c=32, processes=())
        gpus[1] = replace(gpus[1], utilization_gpu_percent=82, memory_used_mb=1024, temperature_c=50)
        gpus[2] = replace(gpus[2], utilization_gpu_percent=15, memory_used_mb=1024, temperature_c=75)
        gpus[3] = replace(gpus[3], utilization_gpu_percent=15, memory_used_mb=14000, memory_total_mb=15109, temperature_c=50)
        gpus[4] = replace(gpus[4], utilization_gpu_percent=15, memory_used_mb=1024, memory_total_mb=15109, temperature_c=85)
        snapshot = replace(snapshot, gpus=tuple(gpus))
        statuses = [TrainingStatus(gpu_index=3, state="stalled")]
        text = render_text(snapshot, statuses=statuses, width=204, height=8)
        for label in ("IDLE", "BUSY", "HOT", "WARN", "CRIT"):
            self.assertIn(label, text)

    def test_auto_wide_short_prioritizes_run_information(self):
        statuses = [
            TrainingStatus(
                pid=12345,
                gpu_index=3,
                run_name="eth_seed7",
                phase="train",
                epoch=4,
                max_epoch=9,
                process_progress_percent=44.0,
                eta_seconds=3600,
                loss=0.82,
                age_seconds=3,
                state="running",
            )
        ]
        text = render_text(snapshot_with_gpus(8), statuses=statuses, width=204, height=8)
        self.assertIn("eth_seed7", text)
        self.assertIn("ETA1h00m", text)
        self.assertIn("HB3s", text)
        self.assertNotIn("┌ G0", text)

    def test_forced_micro_overrides_wide_short_auto(self):
        text = render_text(snapshot_with_gpus(8), width=204, height=8, display_mode="micro", explain_layout=True)
        self.assertIn("layout micro source=forced", text)
        self.assertIn("┌ G0", text)


if __name__ == "__main__":
    unittest.main()
