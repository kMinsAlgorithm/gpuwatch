import importlib.util
import unittest
from unittest.mock import patch

from gpuwatch.sampler import sample_once


@unittest.skipUnless(importlib.util.find_spec("textual"), "Textual is not installed")
class TextualResizeTests(unittest.IsolatedAsyncioTestCase):
    async def test_resize_uses_cached_snapshot_while_paused(self):
        from textual.app import App

        from gpuwatch.render.textual_app import run_textual

        captured = {}
        with patch.object(App, "run", lambda app: captured.setdefault("app", app)):
            run_textual(interval=1, backend="fake", show_training=False)
        app = captured["app"]
        snapshot = sample_once(backend="fake")

        with patch("gpuwatch.render.textual_app.sample_once", return_value=snapshot) as sample:
            async with app.run_test(size=(100, 18)) as pilot:
                await pilot.pause(0.1)
                self.assertEqual(sample.call_count, 1)
                app.action_toggle_pause()
                await pilot.resize_terminal(160, 40)
                await pilot.pause(0.1)
                self.assertEqual(app.query_one("Dashboard").size.width, 160)
                self.assertEqual(sample.call_count, 1)


if __name__ == "__main__":
    unittest.main()
