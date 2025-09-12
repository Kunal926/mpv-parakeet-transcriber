from __future__ import annotations
import math
import re
from typing import List, Dict, Any, Tuple
from segmenter import segment_by_pause_and_phrase, shape_words_into_two_lines_balanced

# Helpers for enforcing minimum readable duration
PUNCT = (".", "!", "?", "…", ",", ":", ";", "-", "—")

__all__ = [
    "enforce_min_readable",
    "postprocess_segments",
    "frame_quantize",
    "write_srt",
    "format_time_srt",
    "srt_ts",
    "normalize_text",
    "pack_into_two_line_blocks",
]


def pack_into_two_line_blocks(
    events: List[Dict[str, Any]],
    max_chars_per_line: int = 40,
    cps_target: float = 20.0,
    coalesce_gap_ms: int = 300,
    two_line_threshold: float = 0.60,
) -> List[Dict[str, Any]]:
    """Greedily merge consecutive events into a single 2-line block when:
      - the inter-event gap ≤ coalesce_gap_ms
      - shaped text fits in 2 lines (no overflow)
      - chars-per-second stays within cps_target
    Preserves word timings; zero text loss."""
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(events):
        blk_words = events[i].get("words") or []
        if not blk_words:
            out.append(events[i])
            i += 1
            continue
        start = float(blk_words[0]["start"])
        end = float(events[i]["end"])
        j = i

        while j + 1 < len(events):
            gap_ms = int(round((events[j + 1]["start"] - events[j]["end"]) * 1000))
            if gap_ms > coalesce_gap_ms:
                break

            nxt_words = events[j + 1].get("words") or []
            cand_words = blk_words + nxt_words
            lines, used_words, overflow = shape_words_into_two_lines_balanced(
                cand_words,
                max_chars_per_line,
                prefer_two_lines=True,
                two_line_threshold=two_line_threshold,
            )
            if overflow:
                break

            cand_text = " ".join(w.get("word", "").strip() for w in cand_words)
            cand_end = float(events[j + 1]["end"])
            cps = len(cand_text.replace("\n", " ").strip()) / max(0.001, (cand_end - start))
            if cps > cps_target:
                break

            blk_words = cand_words
            end = cand_end
            j += 1

        lines, used_words, overflow = shape_words_into_two_lines_balanced(
            blk_words,
            max_chars_per_line,
            prefer_two_lines=True,
            two_line_threshold=two_line_threshold,
        )
        block = {
            "start": float(blk_words[0]["start"]),
            "end": float(end),
            "text": "\n".join(lines[:2]),
            "words": blk_words,
        }
        out.append(block)

        # Practically unreachable because we avoid overflow before packing,
        # but keep it correct just in case.
        while overflow:
            ow = overflow
            lines2, used2, overflow = shape_words_into_two_lines_balanced(
                ow,
                max_chars_per_line,
                prefer_two_lines=True,
                two_line_threshold=two_line_threshold,
            )
            used_block2 = ow[:used2]
            out.append(
                {
                    "start": float(used_block2[0]["start"]),
                    "end": float(used_block2[-1]["end"]),
                    "text": "\n".join(lines2[:2]),
                    "words": used_block2,
                }
            )

        i = j + 1
    return out


def _audit_monotonic_and_lossless(evts):
    # monotonic
    for i in range(1, len(evts)):
        if evts[i]["start"] < evts[i - 1]["end"]:
            print(
                f"[WARN] overlap {i}: {evts[i-1]['end']:.3f} -> {evts[i]['start']:.3f}"
            )
    # lossless text (approximate)
    def flat_text(events):
        return " ".join(e["text"].replace("\n", " ").strip() for e in events)

    return flat_text(evts)


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


def _smart_join(a: str, b: str) -> str:
    """Join two subtitle texts with minimal punctuation artifacts."""
    if not a:
        return b
    if not b:
        return a
    if a.endswith(("-", "—")):
        return (a + b).strip()
    if a.endswith(PUNCT):
        return (a + " " + b).replace("  ", " ").strip()
    return (a + " " + b).replace("  ", " ").strip()


def enforce_min_readable(
    events: list[dict],
    min_dur: float = 0.90,
    max_dur: float = 7.0,
    reflow: bool = True,
    max_chars_per_line: int = 46,
):
    """Extend or merge very short cues so they remain readable."""
    i = 0
    while i < len(events):
        e = events[i]
        dur = e["end"] - e["start"]
        if dur >= min_dur:
            i += 1
            continue

        # 1) try to extend into the gap before the next cue
        if i + 1 < len(events):
            gap = max(0.0, events[i + 1]["start"] - e["end"])
            take = min(min_dur - dur, gap)
            if take > 0:
                e["end"] += take
                dur = e["end"] - e["start"]

        # 2) if still short, merge with closer neighbor
        if dur < min_dur:
            prev_gap = (e["start"] - events[i - 1]["end"]) if i > 0 else 1e9
            next_gap = (events[i + 1]["start"] - e["end"]) if i + 1 < len(events) else 1e9
            if next_gap <= prev_gap and i + 1 < len(events):
                nxt = events[i + 1]
                e["text"] = _smart_join(e["text"], nxt["text"])
                e["end"] = nxt["end"]
                del events[i + 1]
            elif i > 0:
                prv = events[i - 1]
                prv["text"] = _smart_join(prv["text"], e["text"])
                prv["end"] = e["end"]
                del events[i]
                i -= 1
                continue
        i += 1

    if reflow:
        from segmenter import shape_lines_no_loss

        new = []
        for ev in events:
            lines, overflow = shape_lines_no_loss(ev["text"], max_chars_per_line, 2)
            ev["text"] = "\n".join(lines)
            new.append(ev)
            while overflow:
                cont = {"start": ev["end"], "end": ev["end"], "text": overflow}
                lines, overflow = shape_lines_no_loss(cont["text"], max_chars_per_line, 2)
                cont["text"] = "\n".join(lines)
                new.append(cont)
        return new
    return events


def postprocess_segments(
    segments: List[Dict[str, Any]],
    max_chars_per_line: int = 46,
    max_lines: int = 2,
    pause_ms: int = 220,
    cps_target: float = 20.0,
    snap_fps: float | None = None,
    use_spacy: bool = True,
    min_readable: float = 0.9,
    coalesce_gap_ms: int = 300,
    two_line_threshold: float = 0.60,
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
        two_line_threshold=two_line_threshold,
    )

    events = pack_into_two_line_blocks(
        events,
        max_chars_per_line=max_chars_per_line,
        cps_target=cps_target,
        coalesce_gap_ms=coalesce_gap_ms,
        two_line_threshold=two_line_threshold,
    )

    events = enforce_min_readable(
        events,
        min_dur=min_readable,
        max_dur=7.0,
        reflow=True,
        max_chars_per_line=max_chars_per_line,
    )

    _ = _audit_monotonic_and_lossless(events)

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


SPACES = re.compile(r"\s+")
MULTI_DOTS = re.compile(r"\.{3,}")


def normalize_text(t: str) -> str:
    t = t.replace("\u2014", "—").replace("\u2013", "–")
    t = MULTI_DOTS.sub("…", t)
    t = SPACES.sub(" ", t).strip()
    return t
