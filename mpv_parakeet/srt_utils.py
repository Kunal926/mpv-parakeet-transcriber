"""SRT helper utilities."""
from __future__ import annotations

import math

__all__ = ["format_time_srt"]

def format_time_srt(seconds: float) -> str:
    """Convert seconds to ``HH:MM:SS,mmm`` SRT timestamp.

    Values are clamped to zero and non-finite numbers are coerced to ``0.0``.
    Rounding that pushes milliseconds above ``999`` correctly carries to the
    seconds field.
    """
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        seconds = 0.0
    if seconds < 0:
        seconds = 0.0
    total_seconds_int = int(seconds)
    milliseconds = int(round((seconds - total_seconds_int) * 1000))
    if milliseconds >= 1000:
        total_seconds_int += 1
        milliseconds -= 1000
    hours = total_seconds_int // 3600
    minutes = (total_seconds_int % 3600) // 60
    secs = total_seconds_int % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
