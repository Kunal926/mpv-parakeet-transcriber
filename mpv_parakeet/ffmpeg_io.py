"""FFmpeg interaction utilities."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["probe_start", "extract_audio"]


def probe_start(path: Path, ffprobe: str = "ffprobe") -> float:
    """Return the stream start time in seconds using ffprobe.

    Parameters
    ----------
    path: Path
        Media file to inspect.
    ffprobe: str
        Path to ffprobe binary.
    """
    cmd = [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=start_time", "-of", "json", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout or "{}")
    try:
        return float(data["streams"][0]["start_time"])
    except Exception:
        return 0.0


def extract_audio(
    src: Path,
    dst: Path,
    ffmpeg: str = "ffmpeg",
    filters: Optional[str] = None,
    sample_rate: int = 16000,
) -> None:
    """Extract ``src`` audio to mono ``dst`` WAV using ffmpeg."""
    args: list[str] = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
    ]
    if filters:
        args.extend(["-af", filters])
    args.append(str(dst))
    subprocess.run(args, check=True)
