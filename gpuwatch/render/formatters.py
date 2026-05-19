from __future__ import annotations

from typing import Optional

from rich.cells import cell_len, set_cell_size


def na(value: object, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


def percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.0f}%"


def mb(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if value >= 1024:
        return f"{value / 1024.0:.1f}G"
    return f"{value:.0f}M"


def seconds(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    value = int(value)
    hours, rem = divmod(value, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def clamp_text(text: str, width: int) -> str:
    return clamp_cells(text, width, marker="~")


def clamp_cells(text: str, width: int, marker: str = "~") -> str:
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    marker_width = cell_len(marker)
    if width <= marker_width:
        return set_cell_size(text, width)
    return set_cell_size(text, width - marker_width).rstrip() + marker


def bar(value: Optional[float], width: int = 14) -> str:
    if width <= 0:
        return ""
    if value is None:
        return "[" + ("?" * max(1, width - 2)) + "]"
    value = max(0.0, min(100.0, float(value)))
    inner_width = max(1, width - 2)
    filled = int(round((value / 100.0) * inner_width))
    return "[" + ("#" * filled) + ("-" * (inner_width - filled)) + "]"
