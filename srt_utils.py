from __future__ import annotations
import math
import re
from typing import List, Dict, Any, Tuple

# -------------------------
# public API
# -------------------------

def postprocess_segments(
    segments: List[Dict[str, Any]],
    min_duration: float = 5.0/6.0,     # 0.8333 s
    max_duration: float = 7.0,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    min_gap: float = 0.12,             # guard gap between events
    snap_fps: float | None = None,     # e.g. 24.0 -> 1 frame = 41.667 ms
) -> List[Dict[str, Any]]:
    """
    Take raw ASR segments and return SRT-ready events obeying duration,
    line-breaking and positioning rules.
    Each returned dict has: start, end, text (with '\n' between lines).
    """
    # 1) normalize text
    items = []
    for seg in segments:
        if not seg or "text" not in seg:
            continue
        txt = normalize_text(seg["text"])
        if not txt:
            continue
        items.append({
            "start": float(seg.get("start", 0.0)),
            "end":   float(seg.get("end", 0.0)),
            "text":  txt,
            "words": seg.get("words", None),
        })

    # 2) merge/extend to satisfy min_duration
    items = merge_and_pad_short(items, min_duration, min_gap)

    # 3) split anything over max_duration using word or char timing
    items = split_overlong(items, max_duration, min_duration)

    # 4) shape text into at most max_lines obeying break heuristics
    shaped = []
    for it in items:
        lines = shape_lines(it["text"], max_chars_per_line, max_lines)
        it["text"] = "\n".join(lines)
        shaped.append(it)

    # 5) quantize timestamps to fps (optional) and ensure monotonic, gap-safe
    shaped = quantize_and_deoverlap(shaped, snap_fps, min_gap)

    # 6) final sweep: enforce min/max one more time after quantization
    shaped = merge_and_pad_short(shaped, min_duration, min_gap)
    shaped = split_overlong(shaped, max_duration, min_duration)
    shaped = quantize_and_deoverlap(shaped, snap_fps, min_gap)

    return shaped


def write_srt(events: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ev in enumerate(events, 1):
            start = srt_ts(ev["start"])
            end   = srt_ts(ev["end"])
            text  = ev["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


# -------------------------
# internals
# -------------------------

ARTICLES = {"a", "an", "the"}
PRON_SUBJ = {"i","you","he","she","we","they","it"}  # subj. pronouns (heuristic)
CONJ = {"and","but","or","nor","so","yet","for"}
# Common prepositions (not exhaustive, but safe for splitting *before* them)
PREP = {
    "about","above","across","after","against","along","among","around","at","before","behind",
    "below","beneath","beside","between","beyond","by","despite","down","during","except","for",
    "from","in","inside","into","like","near","of","off","on","onto","out","outside","over","past",
    "since","through","throughout","to","toward","under","underneath","until","up","upon","with",
    "within","without"
}
PUNCT_BREAK_AFTER = r"[.!?…:;]"   # break AFTER these

SPACES = re.compile(r"\s+")
MULTI_DOTS = re.compile(r"\.{3,}")  # normalize ellipses
NAME_INITIAL = re.compile(r"^[A-Z]\.$")

def normalize_text(t: str) -> str:
    t = t.replace("\u2014", "—").replace("\u2013", "–")
    t = MULTI_DOTS.sub("…", t)
    t = SPACES.sub(" ", t).strip()
    return t

def merge_and_pad_short(items: List[Dict[str, Any]], min_dur: float, gap: float) -> List[Dict[str, Any]]:
    if not items:
        return items
    out = []
    i = 0
    while i < len(items):
        cur = dict(items[i])
        # Merge forward until min duration satisfied or punctuation boundary reached
        while (cur["end"] - cur["start"] < min_dur) and (i+1 < len(items)):
            nxt = items[i+1]
            # If we can *extend* without overlap, prefer that
            needed_end = cur["start"] + min_dur
            max_end = max(cur["end"], min(nxt["start"] - gap, needed_end))
            if max_end - cur["start"] >= min_dur and max_end <= nxt["start"] - gap:
                cur["end"] = max_end
                break
            # Else merge texts & times
            cur["end"] = max(cur["end"], nxt["end"])
            joiner = " " if needs_space(cur["text"]) else ""
            cur["text"] = (cur["text"] + joiner + nxt["text"]).strip()
            i += 1
        out.append(cur)
        i += 1
    return out

def split_overlong(items: List[Dict[str, Any]], max_dur: float, min_dur: float) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        dur = it["end"] - it["start"]
        if dur <= max_dur + 1e-6:
            out.append(it)
            continue
        # Try to split by word timings first
        words = it.get("words") or []
        if words and all(("start" in w and "end" in w) for w in words):
            out.extend(split_by_words(it, max_dur, min_dur))
        else:
            out.extend(split_by_chars(it, max_dur, min_dur))
    return out

def split_by_words(it: Dict[str, Any], max_dur: float, min_dur: float) -> List[Dict[str, Any]]:
    words = it["words"]
    acc = []
    cur = []
    cur_start = words[0]["start"] if words else it["start"]
    for w in words:
        if cur_start is None:
            cur_start = w["start"]
        cur.append(w)
        # prefer boundary if punctuation ends the word
        cur_end = w["end"]
        dur = cur_end - cur_start
        ends_punct = re.search(PUNCT_BREAK_AFTER, w.get("word","") or "") is not None
        if (dur >= min_dur and (dur >= max_dur or ends_punct)) or dur >= max_dur:
            acc.append(make_item_from_words(cur, cur_start, cur_end))
            cur, cur_start = [], None
    if cur:
        cur_end = cur[-1]["end"]
        if cur_start is None:
            cur_start = it["start"]
        acc.append(make_item_from_words(cur, cur_start, cur_end))
    # if still too long, fall back to char split
    final = []
    for piece in acc:
        if piece["end"] - piece["start"] > max_dur + 1e-6:
            final.extend(split_by_chars(piece, max_dur, min_dur))
        else:
            final.append(piece)
    return final

def make_item_from_words(words, start, end):
    text = " ".join((w.get("word","") or "").strip() for w in words)
    return {"start": float(start), "end": float(end), "text": normalize_text(text)}

def split_by_chars(it: Dict[str, Any], max_dur: float, min_dur: float) -> List[Dict[str, Any]]:
    text = it["text"]
    dur  = it["end"] - it["start"]
    if dur <= 0 or len(text) < 2:
        return [it]
    # time per char heuristic
    tpc = dur / max(1, len(text))
    # number of parts
    n_parts = math.ceil(dur / max_dur)
    part_len = math.ceil(len(text) / n_parts)
    out = []
    s_idx = 0
    s_time = it["start"]
    while s_idx < len(text):
        e_idx = min(len(text), s_idx + part_len)
        # push to a nicer boundary near e_idx
        e_idx = choose_char_boundary(text, e_idx)
        piece = text[s_idx:e_idx].strip()
        p_dur = max(min_dur, min(max_dur, len(piece) * tpc))
        out.append({"start": s_time, "end": s_time + p_dur, "text": piece})
        s_time += p_dur
        s_idx = e_idx
    # ensure last end doesn’t exceed original; if so, compress proportionally
    if out and out[-1]["end"] > it["end"]:
        scale = (it["end"] - out[0]["start"]) / (out[-1]["end"] - out[0]["start"])
        base = out[0]["start"]
        for p in out:
            p["start"] = base + (p["start"] - base) * scale
            p["end"]   = base + (p["end"]   - base) * scale
    return out

def choose_char_boundary(text: str, idx: int) -> int:
    # search window around idx for better split: prefer after punctuation, else at space
    for radius in range(0, 20):
        r = idx + radius
        l = idx - radius
        if r < len(text) and re.match(PUNCT_BREAK_AFTER, text[r-1:r]):
            return r
        if l > 0 and re.match(PUNCT_BREAK_AFTER, text[l-1:l]):
            return l
    # fallback: next space
    rs = text.find(" ", idx)
    if rs != -1:
        return rs
    ls = text.rfind(" ", 0, idx)
    return ls if ls != -1 else idx

def shape_lines(text: str, max_chars: int, max_lines: int) -> List[str]:
    text = normalize_text(text)
    if max_lines <= 1 or len(text) <= max_chars:
        return [text]
    # Two lines maximum
    # Choose the best breakpoint near the middle, following your rules
    words = text.split(" ")
    if len(words) < 2:
        return [text]
    # Try candidate breakpoints at word boundaries
    cut = select_break_index(words, max_chars)
    line1 = " ".join(words[:cut]).strip()
    line2 = " ".join(words[cut:]).strip()
    # If still too long per line, soft-wrap second line within limit (no third line)
    if len(line1) > max_chars and " " in line1:
        line1 = line1[:max_chars+1].rsplit(" ",1)[0]
        # push overflow back to line2
        overflow = text[len(line1):].strip()
        if overflow:
            line2 = overflow
    if len(line2) > max_chars and " " in line2:
        # We cannot add a 3rd line; prefer to keep later words (recency bias)
        line2 = line2[-max_chars:].strip()
    return [line1, line2]

def select_break_index(words: List[str], max_chars: int) -> int:
    # Heuristics: after punctuation; before conjunctions; before prepositions;
    # avoid breaking after articles or subject pronouns; keep names like "J. Smith" together.
    total_len = len(" ".join(words))
    target = total_len // 2

    # score candidates (word index i is the *start* of line2)
    best_i, best_score = 1, -1e9
    for i in range(1, len(words)):
        left = " ".join(words[:i]); right = " ".join(words[i:])
        # hard constraints
        if len(left) > max_chars * 1.25 or len(right) > max_chars * 1.25:
            continue
        prev = words[i-1].strip().strip(",.;:!?…").lower()
        cur  = words[i].strip().strip(",.;:!?…").lower()

        score = 0.0
        # balance around center
        score -= abs(len(left) - target) * 0.5

        # prefer AFTER punctuation
        if re.search(PUNCT_BREAK_AFTER, words[i-1][-1:]):
            score += 6.0

        # prefer BEFORE conjunction or preposition
        if cur in CONJ:
            score += 3.5
        if cur in PREP:
            score += 2.5

        # avoid separating article/adjective from noun (approx: don't break after articles)
        if prev in ARTICLES:
            score -= 6.0

        # avoid breaking after subject pronoun (keeps "I am", "We are")
        if prev in PRON_SUBJ:
            score -= 3.0

        # keep initials with last names (e.g., "J." "Smith")
        if NAME_INITIAL.match(words[i-1]):
            score -= 4.0

        if score > best_score:
            best_score, best_i = score, i

    return max(1, min(best_i, len(words)-1))

def needs_space(t: str) -> bool:
    return len(t) > 0 and not t.endswith((" ", "\n", "-", "—"))

def quantize_and_deoverlap(items: List[Dict[str, Any]], fps: float | None, gap: float) -> List[Dict[str, Any]]:
    if not items:
        return items
    q = 1.0 / fps if fps and fps > 0 else None
    for i, it in enumerate(items):
        if q:
            it["start"] = round(it["start"] / q) * q
            it["end"]   = round(it["end"]   / q) * q
        if i > 0:
            prev = items[i-1]
            if it["start"] < prev["end"] + gap:
                it["start"] = prev["end"] + gap
        if it["end"] <= it["start"]:
            it["end"] = it["start"] + 0.01  # keep strictly increasing
    return items

def srt_ts(t: float) -> str:
    if t < 0: t = 0
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = (int(t) // 3600)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


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

