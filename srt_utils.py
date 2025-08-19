import math


def format_time_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format ``HH:MM:SS,mmm``.

    Any non-numeric, negative, or non-finite (``NaN``/``inf``) values are
    clamped to zero. Rounding that pushes milliseconds to 1000 correctly
    carries over to the seconds component.
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
