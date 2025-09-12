from __future__ import annotations
import math
import re
from typing import List, Dict, Any
from segmenter import segment_by_pause_and_phrase


def postprocess_segments(
    segments: List[Dict[str, Any]],
    max_chars_per_line: int = 40,
    max_lines: int = 2,
    pause_ms: int = 220,
    cps_target: float = 20.0,
    snap_fps: float | None = None,
    use_spacy: bool = True,
) -> List[Dict[str, Any]]:
    """Segment words by pauses and phrases, shape lines, and quantize to frames."""
    # flatten word list (Parakeet provides accurate word timestamps)
    words: List[Dict[str, Any]] = []
    for seg in segments:
        ws = seg.get("words") or []
        for w in ws:
            if "start" in w and "end" in w and w.get("word"):
                words.append({
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })

    events = segment_by_pause_and_phrase(
        words,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        pause_ms=pause_ms,
        cps_target=cps_target,
        use_spacy=use_spacy,
    )

    # frame-quantize: floor starts, ceil ends (prevents late starts)
    events = frame_quantize(events, snap_fps)
    return events


def frame_quantize(items: List[Dict[str, Any]], fps: float | None) -> List[Dict[str, Any]]:
    if not fps:
        return items
    q = 1.0 / fps
    for it in items:
        it["start"] = math.floor(it["start"] / q) * q
        it["end"] = math.ceil(it["end"] / q) * q
        if it["end"] <= it["start"]:
            it["end"] = it["start"] + (1.0 / 100.0)  # 10 ms guard
    # de-overlap without delaying starts; pull previous end back if needed
    for i in range(1, len(items)):
        prev, cur = items[i - 1], items[i]
        if cur["start"] < prev["end"]:
            prev["end"] = min(prev["end"], cur["start"])
    return items


def write_srt(events: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ev in enumerate(events, 1):
            start = format_time_srt(ev["start"])
            end = format_time_srt(ev["end"])
            text = ev["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def format_time_srt(t: float) -> str:
    if not math.isfinite(t) or t < 0:
        t = 0.0
    total_ms = int(round(t * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def srt_ts(t: float) -> str:
    return format_time_srt(t)


SPACES = re.compile(r"\s+")
MULTI_DOTS = re.compile(r"\.{3,}")


def normalize_text(t: str) -> str:
    t = t.replace("\u2014", "—").replace("\u2013", "–")
    t = MULTI_DOTS.sub("…", t)
    t = SPACES.sub(" ", t).strip()
    return t
