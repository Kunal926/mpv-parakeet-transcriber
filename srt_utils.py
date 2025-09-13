from __future__ import annotations
from typing import List, Dict, Any, Tuple
import math, re
from segmenter import segment_by_pause_and_phrase, shape_words_into_two_lines_balanced

SPACES = re.compile(r"\s+")
def normalize_text(t: str) -> str:
    return SPACES.sub(" ", (t or "")).strip()

def srt_ts(t: float) -> str:
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

def format_time_srt(t: float) -> str:
    return srt_ts(t)

def write_srt(events: List[Dict[str,Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ev in enumerate(events, 1):
            f.write(f"{i}\n{srt_ts(ev['start'])} --> {srt_ts(ev['end'])}\n{ev['text'].strip()}\n\n")

# Helper used by timing normalizer to test calm merges for gap chaining
def _can_merge_pair(
    a: Dict[str, Any],
    b: Dict[str, Any],
    max_chars_per_line: int,
    cps_target: float,
    two_line_threshold: float = 0.60,
    min_two_line_chars: int = 24,
):
    from segmenter import shape_words_into_two_lines_balanced

    lw, rw = a.get("words") or [], b.get("words") or []
    if not lw or not rw:
        return False, None
    cand = lw + rw
    lines, used, overflow = shape_words_into_two_lines_balanced(
        cand,
        max_chars=max_chars_per_line,
        prefer_two_lines=True,
        two_line_threshold=two_line_threshold,
        min_two_line_chars=min_two_line_chars,
    )
    if overflow:
        return False, None
    txt = " ".join((w.get("word", "") or "").strip() for w in cand)
    dur = b["end"] - a["start"]
    cps = len(txt) / max(0.001, dur)
    if cps > cps_target:
        return False, None
    return True, (cand, "\n".join(lines[:2]))

# ---------- 2-line packer: merge across short breaths if it fits & cps ok ----------
def pack_into_two_line_blocks(
    events: List[Dict[str, Any]],
    max_chars_per_line: int = 42,
    cps_target: float = 20.0,
    coalesce_gap_ms: int = 360,
    two_line_threshold: float = 0.60,
    min_two_line_chars: int = 24,
) -> List[Dict[str, Any]]:
    out: List[Dict[str,Any]] = []
    i = 0
    while i < len(events):
        blk = events[i]
        bw = blk.get("words") or []
        if not bw:
            out.append(blk); i += 1; continue
        start = float(bw[0]["start"])
        end = float(blk["end"])
        j = i
        while j+1 < len(events):
            gap_ms = int(round((events[j+1]["start"] - events[j]["end"]) * 1000))
            if gap_ms > coalesce_gap_ms: break
            cw = (events[j+1].get("words") or [])
            cand = bw + cw
            lines, used, overflow = shape_words_into_two_lines_balanced(
                cand,
                max_chars=max_chars_per_line,
                prefer_two_lines=True,
                two_line_threshold=two_line_threshold,
                min_two_line_chars=min_two_line_chars,
            )
            if overflow: break
            txt = " ".join((w.get("word","") or "").strip() for w in cand)
            cps = len(txt) / max(0.001, (events[j+1]["end"] - start))
            if cps > cps_target: break
            bw = cand; end = float(events[j+1]["end"]); j += 1
        lines, used, overflow = shape_words_into_two_lines_balanced(
            bw,
            max_chars=max_chars_per_line,
            prefer_two_lines=True,
            two_line_threshold=two_line_threshold,
            min_two_line_chars=min_two_line_chars,
        )
        used_block = bw[:used]
        out.append({
            "start": float(used_block[0]["start"]),
            "end":   float(used_block[-1]["end"]),
            "text":  "\n".join(lines[:2]),
            "words": used_block,
        })
        i = j + 1
    return out

# ---------- orphan-aware min-readable (merges tiny one/two-word singles) ----------
def enforce_min_readable_v2(
    events: List[Dict[str, Any]],
    min_dur: float = 1.10,
    cps_target: float = 20.0,
    max_chars_per_line: int = 42,
    min_two_line_chars: int = 24,
    orphan_words: int = 2,
    orphan_chars: int = 12,
):
    i = 0
    while i < len(events):
        e = events[i]
        dur = e["end"] - e["start"]
        text_flat = e["text"].replace("\n"," ").strip()
        n_words = len(e.get("words") or [])
        is_orphan = (n_words <= orphan_words) or (len(text_flat) <= orphan_chars)
        if dur >= min_dur and not is_orphan:
            i += 1; continue
        # try extend into next gap
        if i+1 < len(events):
            room = max(0.0, events[i+1]["start"] - e["end"])
            need = min_dur - dur
            take = min(need, room)
            if take > 0: e["end"] += take; dur = e["end"] - e["start"]
        if dur >= min_dur and not is_orphan:
            i += 1; continue
        # try merge with prev/next (score by 2-line fit & cps)
        def score_merge(left, right):
            lw, rw = (left.get("words") or []), (right.get("words") or [])
            if not lw or not rw: return -1e9, None
            cand = lw + rw
            lines, used, overflow = shape_words_into_two_lines_balanced(
                cand,
                max_chars=max_chars_per_line,
                prefer_two_lines=True,
                two_line_threshold=0.55,
                min_two_line_chars=min_two_line_chars,
            )
            if overflow: return -1e9, None
            txt = " ".join((w.get("word","") or "").strip() for w in cand)
            cps = len(txt) / max(0.001, (right["end"] - left["start"]))
            if cps > cps_target: return -1e9, None
            diff = abs(len(lines[0]) - len(lines[-1]))
            return -(diff + cps*0.4), (cand, "\n".join(lines[:2]))
        best = None
        if i>0:
            s,p = score_merge(events[i-1], e)
            if p: best=("prev", s, p)
        if i+1 < len(events):
            s,p = score_merge(e, events[i+1])
            if p and (best is None or s>best[1]):
                best=("next", s, p)
        if best:
            which, _, (cand, text) = best
            if which=="prev":
                prev = events[i-1]
                prev["text"] = text
                prev["end"]  = events[i]["end"]
                if prev.get("words") and e.get("words"):
                    prev["words"] = prev["words"] + e["words"]
                del events[i]; i -= 1; continue
            else:
                nxt = events[i+1]
                e["text"] = text
                e["end"]  = nxt["end"]
                if e.get("words") and nxt.get("words"):
                    e["words"] = e["words"] + nxt["words"]
                del events[i+1]; continue
        # last resort: merge orphan forward/back (re-shape; never raw concat)
        if is_orphan:
            from segmenter import shape_words_into_two_lines_balanced
            if i + 1 < len(events):
                nxt = events[i + 1]
                cand_words = (e.get("words") or []) + (nxt.get("words") or [])
                lines, used, overflow = shape_words_into_two_lines_balanced(
                    cand_words,
                    max_chars=max_chars_per_line,
                    prefer_two_lines=True,
                    two_line_threshold=0.60,
                    min_two_line_chars=min_two_line_chars,
                )
                e["text"] = "\n".join(lines[:2])
                e["end"] = nxt["end"]
                e["words"] = cand_words[:used]
                del events[i + 1]
                continue
            elif i > 0:
                prv = events[i - 1]
                cand_words = (prv.get("words") or []) + (e.get("words") or [])
                lines, used, overflow = shape_words_into_two_lines_balanced(
                    cand_words,
                    max_chars=max_chars_per_line,
                    prefer_two_lines=True,
                    two_line_threshold=0.60,
                    min_two_line_chars=min_two_line_chars,
                )
                prv["text"] = "\n".join(lines[:2])
                prv["end"] = e["end"]
                prv["words"] = cand_words[:used]
                del events[i]
                i -= 1
                continue
        i += 1
    return events

# ---------- Netflix timing normalizer ----------
def _spf(fps: float) -> float: return 1.0/max(1e-6, fps)
def _floor(t: float, fps: float) -> float:
    s=_spf(fps); return math.floor(t/s)*s
def _ceil(t: float, fps: float) -> float:
    s=_spf(fps); return math.ceil(t/s)*s

def normalize_timing_netflix(
    events: List[Dict[str,Any]],
    fps: float,
    linger_after_audio_ms: int = 500,
    min_gap_frames: int = 2,
    close_range_frames: Tuple[int,int] = (3,11),  # 24/23.976 only
    small_gap_floor_s: float = 0.5,
    max_chars_per_line: int = 42,
    cps_target: float = 20.0,
    two_line_threshold: float = 0.60,
    min_two_line_chars: int = 24,
) -> List[Dict[str,Any]]:
    if not events: return events
    spf=_spf(fps)
    is_24ish = abs(fps-24.0)<0.2 or abs(fps-23.976)<0.2
    # 1) Start on first audio frame; End on last audio frame (we'll linger later if safe)
    for ev in events:
        ws = ev.get("words") or []
        if ws:
            ev["start"] = _floor(ws[0]["start"], fps)      # never late
            ev["end"]   = _ceil (ws[-1]["end"], fps)
        if ev["end"] <= ev["start"]:
            ev["end"] = ev["start"] + spf
    # 2) Reserve 20 frames for 1–2-word cues; extend longer slightly when space allows
    for i,ev in enumerate(events):
        ws = ev.get("words") or []
        need = 20*spf if len(ws)<=2 else 0.0
        if need and (ev["end"]-ev["start"]) < need:
            if i+1<len(events):
                room = max(0.0, events[i+1]["start"] - ev["end"])
                take = min(need - (ev["end"]-ev["start"]), room)
                if take>0: ev["end"] += take
    # 3) Linger +0.5s ONLY when safe (i.e., there is NOT an immediate next subtitle)
    #    Safe = next.start - last_audio_end >= 0.5s. Otherwise don't linger here.
    for i,ev in enumerate(events):
        ws = ev.get("words") or []
        if not ws: continue
        last_audio_end = ws[-1]["end"]
        if i+1 < len(events):
            gap_to_next = events[i+1]["start"] - last_audio_end
            if gap_to_next >= small_gap_floor_s:
                target = min(
                    last_audio_end + linger_after_audio_ms/1000.0,
                    events[i+1]["start"] - min_gap_frames*spf,
                )
                if target > ev["end"]:
                    ev["end"] = target
        else:
            # last cue in file
            target = last_audio_end + linger_after_audio_ms/1000.0
            if target > ev["end"]:
                ev["end"] = target
    # 4) Chaining / closing gaps
    i = 0
    while i < len(events) - 1:
        a, b = events[i], events[i+1]
        # Snap frame edges for measuring gap
        a["end"] = _ceil(a["end"], fps)
        b["start"] = _floor(b["start"], fps)

        spf = _spf(fps)
        gap_s = b["start"] - a["end"]
        gap_f = int(round(gap_s / spf))

        # Audio boundaries
        a_ws = a.get("words") or []
        a_audio_end = _ceil(a_ws[-1]["end"], fps) if a_ws else a["end"]
        max_linger = a_audio_end + linger_after_audio_ms/1000.0

        # If gap < 2f:
        if gap_f < min_gap_frames:
            # Try borrowing time by merging into a calm 2-line block
            ok, payload = _can_merge_pair(
                a,
                b,
                max_chars_per_line=max_chars_per_line,
                cps_target=cps_target,
                two_line_threshold=two_line_threshold,
                min_two_line_chars=min_two_line_chars,
            )
            if ok and payload:
                words, text = payload
                a["text"] = text
                a["end"] = b["end"]
                if a.get("words") and b.get("words"):
                    a["words"] = a["words"] + b["words"]
                del events[i+1]
                # do not advance i; new a may chain with the next b
                continue

            # Otherwise, chain only if it does NOT cut into A audio
            desired = b["start"] - min_gap_frames * spf
            desired = min(desired, max_linger)
            if desired >= a_audio_end:
                a["end"] = desired
            else:
                # Preserve A audio: accept <2f gap (do NOT delay B in)
                a["end"] = max(a["end"], a_audio_end)

            if a["end"] <= a["start"]:
                a["end"] = a["start"] + spf

        else:
            # 24ish: close specified range to 2f, but still never cut earlier than audio
            is_24ish = abs(fps - 24.0) < 0.2 or abs(fps - 23.976) < 0.2
            low, high = close_range_frames
            if is_24ish and (low <= gap_f <= high):
                desired = b["start"] - min_gap_frames * spf
                desired = min(desired, max_linger)
                a["end"] = max(a_audio_end, desired)
            else:
                # general rule: either 2f or ≥0.5s
                if gap_s < small_gap_floor_s:
                    desired = b["start"] - min_gap_frames * spf
                    desired = min(desired, max_linger)
                    a["end"] = max(a_audio_end, desired)

        # never invert
        if a["end"] <= a["start"]:
            a["end"] = a["start"] + spf
        i += 1
    # 5) final snap & monotonic
    for i,ev in enumerate(events):
        ev["start"] = _floor(ev["start"], fps)
        ev["end"]   = _ceil (ev["end"], fps)
        if i>0 and ev["start"] < events[i-1]["end"] + min_gap_frames*spf:
            ev["start"] = events[i-1]["end"] + min_gap_frames*spf
        if ev["end"] <= ev["start"]:
            ev["end"] = ev["start"] + spf
    return events

# ---------- top-level postprocess ----------
def postprocess_segments(
    segments: List[Dict[str,Any]],
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    pause_ms: int = 240,
    punct_pause_ms: int = 160,
    comma_pause_ms: int = 120,
    cps_target: float = 20.0,
    snap_fps: float | None = None,
    use_spacy: bool = True,
    coalesce_gap_ms: int = 360,
    two_line_threshold: float = 0.60,
    min_readable: float = 1.10,
    min_two_line_chars: int = 24,
) -> List[Dict[str,Any]]:
    # Flatten word list from raw ASR segments
    words: List[Dict[str,Any]] = []
    for seg in segments:
        for w in (seg.get("words") or []):
            if "start" in w and "end" in w and w.get("word"):
                words.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
    # Segment on pauses/phrases, shape to 2 lines (word-preserving)
    events = segment_by_pause_and_phrase(
        words,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        pause_ms=pause_ms,
        punct_pause_ms=punct_pause_ms,
        comma_pause_ms=comma_pause_ms,
        cps_target=cps_target,
        use_spacy=use_spacy,
        two_line_threshold=two_line_threshold,
        min_two_line_chars=min_two_line_chars,
    )
    # Merge small neighbors into calm 2-line blocks
    events = pack_into_two_line_blocks(
        events,
        max_chars_per_line=max_chars_per_line,
        cps_target=cps_target,
        coalesce_gap_ms=coalesce_gap_ms,
        two_line_threshold=two_line_threshold,
        min_two_line_chars=min_two_line_chars,
    )
    # Eliminate quick singles (orphans) and short flashes
    events = enforce_min_readable_v2(
        events,
        min_dur=min_readable,
        cps_target=cps_target,
        max_chars_per_line=max_chars_per_line,
        min_two_line_chars=min_two_line_chars,
    )
    # Netflix timing: linger-only-when-safe + chaining + 20f for 1–2 words
    if snap_fps:
        events = normalize_timing_netflix(
            events,
            fps=snap_fps,
            linger_after_audio_ms=500,
            min_gap_frames=2,
            close_range_frames=(3,11),
            small_gap_floor_s=0.5,
            max_chars_per_line=max_chars_per_line,
            cps_target=cps_target,
            two_line_threshold=two_line_threshold,
            min_two_line_chars=min_two_line_chars,
        )
    return events
