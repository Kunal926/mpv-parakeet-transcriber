from __future__ import annotations
from typing import List, Dict, Any, Tuple
import math, sys, os, time, re, csv, json
from segmenter import segment_by_pause_and_phrase, shape_words_into_two_lines_balanced

# ---------- tiny helpers ----------
def _trace(msg: str) -> None:
    if os.environ.get("PARAKEET_TRACE_TIMING") == "1":
        sys.stderr.write(f"[srt_utils] {msg}\n"); sys.stderr.flush()

def _with_timeout(timeout_s: float, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) in a daemon thread; return result or None on timeout."""
    import threading, copy
    out, err = {}, []
    if args and isinstance(args[0], list):
        args = (copy.deepcopy(args[0]),) + args[1:]
    done = threading.Event()
    def _run():
        try:
            out["v"] = fn(*args, **kwargs)
        except Exception as e:
            err.append(e)
        finally:
            done.set()
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if not done.wait(timeout_s):
        _trace(f"{getattr(fn, '__name__', 'func')} timed out after {timeout_s:.1f}s — skipping")
        return None
    if err:
        e = err[0]
        if isinstance(e, AssertionError):
            _trace(f"{getattr(fn, '__name__', 'func')} failed: {e}")
            return None
        raise e
    return out.get("v")

SPACES = re.compile(r"\s+")
def normalize_text(t: str) -> str:
    return SPACES.sub(" ", (t or "")).strip()

# --- diagnostics helpers ---
def _dbg(ev):
    d = ev.get("_dbg")
    if d is None:
        d = {}
        ev["_dbg"] = d
    return d

def _dbg_add_ms(ev, key, ms):
    if ms <= 0:
        return
    d = _dbg(ev)
    d[key] = int(d.get(key, 0) + round(ms))

def _last_word_end(ev):
    w = ev.get("words") or []
    return float(w[-1]["end"]) if w else float(ev["end"])

def _final_linger_ms(ev) -> int:
    audio_end = _last_word_end(ev)
    return max(0, int(round((ev["end"] - audio_end) * 1000)))

def _chars(ev):
    """Character count of rendered text (shaped with line breaks removed)."""
    return len((ev.get("text") or "").replace("\n", ""))

def _ms_floor(t: float) -> int:
    return 0 if not math.isfinite(t) or t < 0 else math.floor(t * 1000 + 1e-9)

def _ms_ceil(t: float) -> int:
    return 0 if not math.isfinite(t) or t < 0 else math.ceil(t * 1000 - 1e-9)

def _fmt_ms(total_ms: int) -> str:
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def format_start_ms(t: float) -> str: return _fmt_ms(_ms_ceil (t))
def format_end_ms  (t: float) -> str: return _fmt_ms(_ms_floor(t))

def _write_diag_sidecar(events, out_path, diag_dir: str | None = None):
    rows = []
    for i, ev in enumerate(events, 1):
        d = ev.get("_dbg", {}) or {}
        dur_f = max(1e-9, ev["end"] - ev["start"])
        cps_f = _chars(ev) / dur_f
        s_ms = _ms_ceil(ev["start"])
        e_ms = _ms_floor(ev["end"])
        dur_ms_render = max(0, e_ms - s_ms)
        rows.append({
            "idx": i,
            "start": ev["start"],
            "end": ev["end"],
            "dur_ms": int(round(dur_f * 1000)),
            "cps_float": round(cps_f, 2),
            "dur_ms_rendered": dur_ms_render,
            "linger_ms": _final_linger_ms(ev),
            "linger_added_ms": int(d.get("linger_ms", 0)),
            "linger_clamped": bool(d.get("linger_clamped", False)),
            "borrow_from_right_ms": d.get("borrow_from_right_ms", 0),
            "gave_to_left_ms": d.get("gave_to_left_ms", 0),
            "final_gap_fence_ms": d.get("final_gap_fence_ms", 0),
        })
    # If a diagnostics directory is provided, write the sidecars there,
    # using the SRT basename (without extension) to keep filenames readable.
    if diag_dir:
        os.makedirs(diag_dir, exist_ok=True)
        base = os.path.join(diag_dir, os.path.splitext(os.path.basename(out_path))[0])
    else:
        base = out_path
    with open(base + ".diag.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(base + ".diag.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def write_srt(events: List[Dict[str,Any]], out_path: str, diag_dir: str | None = None) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ev in enumerate(events, 1):
            f.write(f"{i}\n{format_start_ms(ev['start'])} --> {format_end_ms(ev['end'])}\n{(ev['text'] or '').strip()}\n\n")
    try:
        _write_diag_sidecar(events, out_path, diag_dir)
    except Exception:
        pass

# helper: cps of an event
def _cps_of(ev: Dict[str, Any]) -> float:
    txt = " ".join((w.get("word", "") or "").strip() for w in (ev.get("words") or []))
    dur = max(0.001, ev["end"] - ev["start"])
    return len(txt) / dur

def _merged_words(a: Dict[str, Any], b: Dict[str, Any]):
    return (a.get("words") or []) + (b.get("words") or [])

# helper: can two cues be safely merged into a single 2-line block?
def _can_merge_pair(
    a: Dict[str,Any],
    b: Dict[str,Any],
    max_chars_per_line: int,
    cps_target: float,
    two_line_threshold: float = 0.60,
    min_two_line_chars: int = 24,
    max_block_duration_s: float | None = None,
    shaper=shape_words_into_two_lines_balanced,
    allow_cps_decrease: bool = False,
):
    lw, rw = (a.get("words") or []), (b.get("words") or [])
    if not lw or not rw:
        return False, None
    cand = _merged_words(a, b)
    lines, used, overflow = shaper(
        cand,
        max_chars=max_chars_per_line,
        prefer_two_lines=True,
        two_line_threshold=two_line_threshold,
        min_two_line_chars=min_two_line_chars,
    )
    if overflow:
        return False, None
    txt = " ".join((w.get("word","") or "").strip() for w in cand)
    dur = b["end"] - a["start"]
    cps_merged = len(txt) / max(0.001, dur)
    if max_block_duration_s is not None and dur > max_block_duration_s:
        return False, None
    if cps_merged <= cps_target:
        return True, (cand, "\n".join(lines[:2]))
    if allow_cps_decrease:
        worst_local = max(_cps_of(a), _cps_of(b))
        if cps_merged < worst_local - 1e-6:
            return True, (cand, "\n".join(lines[:2]))
    return False, None

# ---------- 2-line packer: merge across short breaths if it fits & cps ok ----------
def pack_into_two_line_blocks(
    events: List[Dict[str, Any]],
    max_chars_per_line: int = 42,
    cps_target: float = 20.0,
    coalesce_gap_ms: int = 360,
    two_line_threshold: float = 0.60,
    min_two_line_chars: int = 24,
    max_block_duration_s: float = 7.0,
    shaper=shape_words_into_two_lines_balanced,
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
        tries = 0
        MAX_TRIES = 32  # hard cap per block
        while j+1 < len(events) and tries < MAX_TRIES:
            gap_ms = int(round((events[j+1]["start"] - events[j]["end"]) * 1000))
            if gap_ms > coalesce_gap_ms: break
            # Do not let the candidate block grow beyond max duration
            cand_end = float(events[j+1]["end"])
            if (cand_end - start) > max_block_duration_s:
                break
            cw = (events[j+1].get("words") or [])
            # --- Cheap prefilters before expensive shaping ---
            if not cw:
                break
            # rough char count (no spaces). Allow small slack.
            naive_chars = sum(len((w.get("word","").strip())) for w in (bw+cw))
            if naive_chars > (max_chars_per_line*2 + 8):
                break
            naive_dur = (events[j+1]["end"] - start) or 1e-3
            if (naive_chars / naive_dur) > (cps_target * 1.15):
                break
            # -------------------------------------------------
            cand = bw + cw
            lines, used, overflow = shaper(
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
            tries += 1
        lines, used, overflow = shaper(
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
    max_merge_gap_ms: int = 360,
    orphan_words: int = 2,
    orphan_chars: int = 12,
    shaper=shape_words_into_two_lines_balanced,
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
        def score_merge(left, right, _shaper=shaper):
            lw, rw = (left.get("words") or []), (right.get("words") or [])
            if not lw or not rw:
                return -1e9, None
            # Respect gap limit for borrowing/merge
            gap_ms = int(round((right["start"] - left["end"]) * 1000))
            if gap_ms > max_merge_gap_ms:
                return -1e9, None
            cand = lw + rw
            lines, used, overflow = _shaper(
                cand,
                max_chars=max_chars_per_line,
                prefer_two_lines=True,
                two_line_threshold=0.60,
                min_two_line_chars=min_two_line_chars,
            )
            if overflow:
                return -1e9, None
            txt = " ".join((w.get("word", "") or "").strip() for w in cand)
            dur = max(0.001, right["end"] - left["start"])
            cps = len(txt) / dur
            # New policy: allow merge when merged CPS strictly decreases
            # versus the worst local CPS, even if still above target.
            if cps > cps_target:
                def _cps_of(ev):
                    t = (ev.get("text") or "")
                    if not t:
                        t = " ".join((w.get("word", "") or "").strip() for w in (ev.get("words") or []))
                    d = max(0.001, ev["end"] - ev["start"])
                    return len(t.replace("\n", " ")) / d
                worst_local = max(_cps_of(left), _cps_of(right))
                # Require a strictly lower CPS (epsilon for float noise)
                if cps >= worst_local - 1e-6:
                    return -1e9, None
            diff = abs(len(lines[0]) - len(lines[-1]))
            return -(diff + cps * 0.4), (cand, "\n".join(lines[:2]))
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
                prev["words"] = _merged_words(prev, e)
                del events[i]; i -= 1; continue
            else:
                nxt = events[i+1]
                e["text"] = text
                e["end"]  = nxt["end"]
                e["words"] = _merged_words(e, nxt)
                del events[i+1]; continue
        # last resort: merge orphan forward/back (re-shape; never raw concat)
        if is_orphan:
            if i + 1 < len(events):
                nxt = events[i+1]
                # Only merge forward if the gap is small
                gap_ms = int(round((nxt["start"] - e["end"]) * 1000))
                if gap_ms <= max_merge_gap_ms:
                    cand_words = _merged_words(e, nxt)
                    lines, used, overflow = shaper(
                        cand_words,
                        max_chars=max_chars_per_line,
                        prefer_two_lines=True,
                        two_line_threshold=0.60,
                        min_two_line_chars=min_two_line_chars,
                    )
                    used_block = cand_words[:used]
                    e["text"]  = "\n".join(lines[:2])
                    e["end"]   = float(used_block[-1]["end"])
                    e["words"] = used_block
                    # If overflow exists, emit it as its own event(s)
                    k = i+1
                    cur_over = overflow
                    guard = 0
                    while cur_over and guard < 1000:
                        lines2, used2, over2 = shaper(
                            cur_over,
                            max_chars=max_chars_per_line,
                            prefer_two_lines=True,
                            two_line_threshold=0.60,
                            min_two_line_chars=min_two_line_chars,
                        )
                        # Guarantee progress even on pathological tokens (e.g., 60+ char word)
                        if used2 <= 0:
                            used2 = 1
                            lines2 = [normalize_text(cur_over[0].get("word",""))]
                            over2  = cur_over[1:]
                        used_block2 = cur_over[:used2]
                        events.insert(k, {
                            "start": float(used_block2[0]["start"]),
                            "end":   float(used_block2[-1]["end"]),
                            "text":  "\n".join(lines2[:2]),
                            "words": used_block2,
                        })
                        k += 1
                        cur_over = over2
                        guard += 1
                    # Remove the original neighbor we merged into
                    del events[k]
                    continue
                # else: cannot merge across long gap; fall through
            elif i > 0:
                prev = events[i-1]
                # Only merge backward if the gap is small
                gap_ms = int(round((e["start"] - prev["end"]) * 1000))
                if gap_ms <= max_merge_gap_ms:
                    cand_words = _merged_words(prev, e)
                    lines, used, overflow = shaper(
                        cand_words,
                        max_chars=max_chars_per_line,
                        prefer_two_lines=True,
                        two_line_threshold=0.60,
                        min_two_line_chars=min_two_line_chars,
                    )
                    used_block = cand_words[:used]
                    prev["text"]  = "\n".join(lines[:2])
                    prev["end"]   = float(used_block[-1]["end"])
                    prev["words"] = used_block
                    # If overflow exists, emit it as its own event(s)
                    k = i
                    cur_over = overflow
                    guard = 0
                    while cur_over and guard < 1000:
                        lines2, used2, over2 = shaper(
                            cur_over,
                            max_chars=max_chars_per_line,
                            prefer_two_lines=True,
                            two_line_threshold=0.60,
                            min_two_line_chars=min_two_line_chars,
                        )
                        # Guarantee progress even on pathological tokens (e.g., 60+ char word)
                        if used2 <= 0:
                            used2 = 1
                            lines2 = [normalize_text(cur_over[0].get("word",""))]
                            over2  = cur_over[1:]
                        used_block2 = cur_over[:used2]
                        events.insert(k, {
                            "start": float(used_block2[0]["start"]),
                            "end":   float(used_block2[-1]["end"]),
                            "text":  "\n".join(lines2[:2]),
                            "words": used_block2,
                        })
                        k += 1
                        cur_over = over2
                        guard += 1
                    # Remove the orphan we merged
                    del events[k]
                    i -= 1
                    continue
                # else: cannot merge across long gap; fall through
            # If we get here, we couldn't merge (gap too big) — we already tried extending forward in step 1.
            # Leave as-is; the normalizer will apply safe +0.5s linger if applicable.
            i += 1
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
    shaper=shape_words_into_two_lines_balanced,
    max_block_duration_s: float = 7.0,
    validate: bool = True,
) -> List[Dict[str,Any]]:
    if not events:
        return events
    for ev in events:
        dbg = ev.get("_dbg")
        if dbg:
            dbg.pop("linger_ms", None)
            dbg.pop("linger_clamped", None)
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
    # 2) Enforce 20-frame minimum duration by extending or merging
    i = 0
    min_dur = 20*spf
    while i < len(events):
        ev = events[i]
        dur = ev["end"] - ev["start"]
        if dur < min_dur:
            # Try merge forward if gap small and mergeable
            if i + 1 < len(events):
                gap_to_next = events[i + 1]["start"] - ev["end"]
                if gap_to_next <= small_gap_floor_s:
                    ok, payload = _can_merge_pair(
                        ev, events[i + 1],
                        max_chars_per_line, cps_target,
                        two_line_threshold, min_two_line_chars,
                        max_block_duration_s, shaper,
                        allow_cps_decrease=True,
                    )
                    if ok and payload:
                        words, text = payload
                        ev["text"] = text
                        ev["end"] = events[i + 1]["end"]
                        ev["words"] = _merged_words(ev, events[i + 1])
                        del events[i + 1]
                        dur = ev["end"] - ev["start"]
                        # re-evaluate same index after merge
                        continue
            # If forward merge failed, try merge backward next
            if i > 0:
                gap_to_prev = ev["start"] - events[i - 1]["end"]
                if gap_to_prev <= small_gap_floor_s:
                    ok, payload = _can_merge_pair(
                        events[i - 1], ev,
                        max_chars_per_line, cps_target,
                        two_line_threshold, min_two_line_chars,
                        max_block_duration_s, shaper,
                        allow_cps_decrease=True,
                    )
                    if ok and payload:
                        prev = events[i - 1]
                        words, text = payload
                        prev["text"] = text
                        prev["end"] = ev["end"]
                        prev["words"] = _merged_words(prev, ev)
                        del events[i]
                        i -= 1
                        continue
            # Extend towards next cue if room
            if i + 1 < len(events):
                max_end = events[i + 1]["start"] - min_gap_frames * spf
                target = min(ev["start"] + min_dur, max_end)
                if target > ev["end"]:
                    ev["end"] = target
                    dur = ev["end"] - ev["start"]
            # If still short, try merge backward again
            if dur < min_dur and i > 0:
                gap_to_prev = ev["start"] - events[i - 1]["end"]
                if gap_to_prev <= small_gap_floor_s:
                    ok, payload = _can_merge_pair(
                        events[i - 1], ev,
                        max_chars_per_line, cps_target,
                        two_line_threshold, min_two_line_chars,
                        max_block_duration_s, shaper,
                        allow_cps_decrease=True,
                    )
                    if ok and payload:
                        prev = events[i - 1]
                        words, text = payload
                        prev["text"] = text
                        prev["end"] = ev["end"]
                        prev["words"] = _merged_words(prev, ev)
                        del events[i]
                        i -= 1
                        continue
        i += 1
    # 3) Linger +0.5s ONLY when safe (i.e., there is NOT an immediate next subtitle)
    #    Safe = next.start - last_audio_end >= 0.5s. Otherwise don't linger here.
    for i,ev in enumerate(events):
        ws = ev.get("words") or []
        if not ws:
            continue
        audio_end = _last_word_end(ev)
        target = ev["end"]
        if i+1 < len(events):
            gap_to_next = events[i+1]["start"] - audio_end
            if gap_to_next >= small_gap_floor_s:
                candidate = min(
                    audio_end + linger_after_audio_ms/1000.0,
                    events[i+1]["start"] - min_gap_frames*spf,
                )
                if candidate > ev["end"]:
                    target = candidate
        else:
            # last cue in file
            candidate = audio_end + linger_after_audio_ms/1000.0
            if candidate > ev["end"]:
                target = candidate
        prev_end = ev["end"]
        ev["end"] = target
        _linger_ms = max(0.0, (target - audio_end) * 1000.0)
        _dbg_add_ms(ev, "linger_ms", _linger_ms)
        if target < audio_end + linger_after_audio_ms/1000.0 - 1e-9:
            _dbg(ev)["linger_clamped"] = True
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
        b_ws = b.get("words") or []
        b_audio_floor = _floor(b_ws[0]["start"], fps) if b_ws else b["start"]

        if gap_f < min_gap_frames:
            # Borrow time first (merge) if it yields a valid 2-line block within cps AND cap
            ok, payload = _can_merge_pair(
                a, b,
                max_chars_per_line, cps_target,
                two_line_threshold, min_two_line_chars,
                max_block_duration_s, shaper,
                allow_cps_decrease=True,
            )
            if ok and payload:
                words, text = payload
                a["text"] = text
                a["end"] = b["end"]
                a["words"] = _merged_words(a, b)
                del events[i+1]
                continue
            # Else, resolve without overlap
            latest_b = b_audio_floor + min_gap_frames*spf
            if b["start"] < a["end"]:
                b["start"] = min(max(a["end"], b["start"]), latest_b)
                gap_s = b["start"] - a["end"]
                gap_f = int(round(gap_s / spf))
            desired = b["start"] - min_gap_frames*spf
            desired = min(desired, max_linger)
            if desired >= a_audio_end:
                a["end"] = desired
            else:
                a["end"] = max(a["end"], a_audio_end)
            if a["end"] <= a["start"]:
                a["end"] = a["start"] + spf
        else:
            is_24ish = abs(fps - 24.0) < 0.2 or abs(fps - 23.976) < 0.2
            low, high = close_range_frames
            if is_24ish and (low <= gap_f <= high):
                desired = b["start"] - min_gap_frames*spf
                desired = min(desired, max_linger)
                a["end"] = max(a_audio_end, desired)
            else:
                if gap_s < small_gap_floor_s:
                    desired = b["start"] - min_gap_frames*spf
                    desired = min(desired, max_linger)
                    a["end"] = max(a_audio_end, desired)
        if a["end"] <= a["start"]:
            a["end"] = a["start"] + spf
        i += 1
    # 5) final snap & monotonic (safe clamp, never late beyond audio +2f)
    for i, ev in enumerate(events):
        ev["start"] = _floor(ev["start"], fps)
        ev["end"] = _ceil(ev["end"], fps)
        if i > 0:
            ws = ev.get("words") or []
            audio_start_floor = _floor(ws[0]["start"], fps) if ws else ev["start"]
            lower = events[i-1]["end"] + min_gap_frames*spf
            upper = audio_start_floor + min_gap_frames*spf
            if upper < lower:
                upper = lower
            ev["start"] = min(max(ev["start"], lower), upper)
        if ev["end"] <= ev["start"]:
            ev["end"] = ev["start"] + spf

    # 6) post-quantization gap normalizer (force ≥2f whenever gap < floor)
    for i in range(len(events) - 1):
        gap = events[i+1]["start"] - events[i]["end"]
        if gap < small_gap_floor_s:
            events[i]["end"] = events[i+1]["start"] - min_gap_frames*spf
            if events[i]["end"] <= events[i]["start"]:
                events[i]["end"] = events[i]["start"] + spf

    # helpers for duration and CPS borrowing
    def _borrow_from_right(idx: int, need: float, tolerance: float = 0.002) -> bool:
        if idx + 1 >= len(events):
            return False
        cur, nxt = events[idx], events[idx + 1]
        gap = nxt["start"] - cur["end"]
        spare_gap = max(0.0, gap - min_gap_frames*spf)
        spare_nxt = max(0.0, (nxt["end"] - nxt["start"]) - min_dur)
        avail = spare_gap + spare_nxt
        if avail <= 0:
            return False
        borrow = min(need, avail)
        orig_end = cur["end"]
        orig_start_nxt = nxt["start"]
        orig_end_nxt = nxt["end"]
        give = 0.0
        take = min(spare_gap, borrow)
        # use silent gap without shifting the next cue's start
        cur["end"] += take
        borrow = max(0.0, borrow - take)
        if borrow > 0:
            cur["end"] += borrow
            nxt["start"] += borrow
            give = borrow
            ws = nxt.get("words") or []
            if ws:
                audio_start = _floor(ws[0]["start"], fps)
                upper = audio_start + min_gap_frames*spf
                if nxt["start"] > upper:
                    shift = nxt["start"] - upper
                    nxt["start"] = upper
                    cur["end"] -= shift
            # try to restore nxt duration so CPS stays stable
            ws2 = nxt.get("words") or []
            nxt_audio_end = _ceil(ws2[-1]["end"], fps) if ws2 else orig_end_nxt
            next_of_next_start = events[idx + 2]["start"] if idx + 2 < len(events) else float("inf")
            linger_cap = nxt_audio_end + linger_after_audio_ms/1000.0
            max_extend = min(linger_cap, next_of_next_start - min_gap_frames*spf) - orig_end_nxt
            if max_extend > 0:
                nxt["end"] = min(orig_end_nxt + borrow, orig_end_nxt + max_extend)
        dur_nxt = nxt["end"] - nxt["start"]
        txt_nxt = " ".join((w.get("word", "") or "").strip() for w in nxt.get("words") or [])
        cps_nxt = len(txt_nxt) / max(0.001, dur_nxt)
        changed = (cur["end"] != orig_end or nxt["start"] != orig_start_nxt or nxt["end"] != orig_end_nxt)
        if dur_nxt + tolerance < min_dur or cps_nxt > cps_target or not changed:
            cur["end"] = orig_end
            nxt["start"] = orig_start_nxt
            nxt["end"] = orig_end_nxt
            return False
        _dbg_add_ms(cur, "borrow_from_right_ms", give * 1000.0)
        _dbg_add_ms(nxt, "gave_to_left_ms", give * 1000.0)
        return True

    def _borrow_from_left(idx: int, need: float, tolerance: float = 0.002) -> bool:
        if idx == 0:
            return False
        prev, cur = events[idx - 1], events[idx]
        gap = cur["start"] - prev["end"]
        spare_gap = max(0.0, gap - min_gap_frames*spf)

        # respect each block's audio boundaries
        cur_ws = cur.get("words") or []
        prev_ws = prev.get("words") or []
        cur_audio_start = _floor(cur_ws[0]["start"], fps) if cur_ws else cur["start"]
        prev_audio_end = _ceil(prev_ws[-1]["end"], fps) if prev_ws else prev["end"]
        spare_cur = max(0.0, cur["start"] - cur_audio_start)
        earliest_prev_end = max(prev_audio_end, prev["start"] + min_dur)
        spare_prev = max(0.0, prev["end"] - earliest_prev_end)

        avail = min(spare_gap + spare_prev, spare_cur)
        if avail <= 0:
            return False
        borrow = min(need, avail)
        orig_start = cur["start"]
        orig_end_prev = prev["end"]
        take = min(spare_gap, borrow)
        # pull from available gap without moving the previous cue's end
        cur["start"] -= take
        borrow = max(0.0, borrow - take)
        if borrow > 0:
            cur["start"] -= borrow
            prev["end"] -= borrow

        dur_prev = prev["end"] - prev["start"]
        txt_prev = " ".join((w.get("word", "") or "").strip() for w in prev.get("words") or [])
        cps_prev = len(txt_prev) / max(0.001, dur_prev)
        # ensure we didn't trim past audio end or violate min dur/cps
        if prev["end"] < prev_audio_end - tolerance or dur_prev + tolerance < min_dur or cps_prev > cps_target:
            cur["start"] = orig_start
            prev["end"] = orig_end_prev
            return False
        # final safeguard: don't start before audio
        if cur["start"] < cur_audio_start:
            diff = cur_audio_start - cur["start"]
            cur["start"] = cur_audio_start
            prev["end"] -= diff
            if prev["end"] < prev_audio_end - tolerance:
                cur["start"] = orig_start
                prev["end"] = orig_end_prev
                return False
        return True

    # 7) duration repair (ensure ≥20f)
    tolerance = 0.002
    min_dur = 20 * spf
    i = 0
    while i < len(events):
        ev = events[i]
        dur = ev["end"] - ev["start"]
        if dur + tolerance < min_dur:
            need = min_dur - dur
            if _borrow_from_right(i, need):
                continue
            if _borrow_from_left(i, need):
                continue
            merged = False
            if i + 1 < len(events):
                ok, payload = _can_merge_pair(
                    ev,
                    events[i + 1],
                    max_chars_per_line,
                    cps_target,
                    two_line_threshold,
                    min_two_line_chars,
                    max_block_duration_s,
                    shaper,
                    allow_cps_decrease=True,
                )
                if ok and payload:
                    words, text = payload
                    ev["text"] = text
                    ev["end"] = events[i + 1]["end"]
                    ev["words"] = _merged_words(ev, events[i + 1])
                    del events[i + 1]
                    merged = True
            if not merged and i > 0:
                ok, payload = _can_merge_pair(
                    events[i - 1],
                    ev,
                    max_chars_per_line,
                    cps_target,
                    two_line_threshold,
                    min_two_line_chars,
                    max_block_duration_s,
                    shaper,
                    allow_cps_decrease=True,
                )
                if ok and payload:
                    prev = events[i - 1]
                    words, text = payload
                    prev["text"] = text
                    prev["end"] = ev["end"]
                    prev["words"] = _merged_words(prev, ev)
                    del events[i]
                    i -= 1
                    continue
        i += 1

    # 8) reading speed balance (borrow before merge)
    i = 0
    while i < len(events):
        ev = events[i]
        txt = " ".join((w.get("word", "") or "").strip() for w in ev.get("words") or [])
        dur = ev["end"] - ev["start"]
        cps = len(txt) / max(0.001, dur)
        if cps > cps_target:
            need = (len(txt) / cps_target) - dur
            if _borrow_from_right(i, need):
                continue
            if _borrow_from_left(i, need):
                continue
            merged = False
            if i + 1 < len(events):
                ok, payload = _can_merge_pair(
                    ev,
                    events[i + 1],
                    max_chars_per_line,
                    cps_target,
                    two_line_threshold,
                    min_two_line_chars,
                    max_block_duration_s,
                    shaper,
                    allow_cps_decrease=True,
                )
                if ok and payload:
                    words, text = payload
                    ev["text"] = text
                    ev["end"] = events[i + 1]["end"]
                    ev["words"] = _merged_words(ev, events[i + 1])
                    del events[i + 1]
                    continue
            if i > 0:
                ok, payload = _can_merge_pair(
                    events[i - 1],
                    ev,
                    max_chars_per_line,
                    cps_target,
                    two_line_threshold,
                    min_two_line_chars,
                    max_block_duration_s,
                    shaper,
                    allow_cps_decrease=True,
                )
                if ok and payload:
                    prev = events[i - 1]
                    words, text = payload
                    prev["text"] = text
                    prev["end"] = ev["end"]
                    prev["words"] = _merged_words(prev, ev)
                    del events[i]
                    i -= 1
                    continue
        i += 1

    # 9) final snap & validate
    for ev in events:
        ev["start"] = _floor(ev["start"], fps)
        ev["end"] = _ceil(ev["end"], fps)
        if ev["end"] <= ev["start"]:
            ev["end"] = ev["start"] + spf

    do_validate = validate

    def _enforce(cond: bool, msg: str) -> None:
        if do_validate:
            assert cond, msg
        elif not cond:
            _trace(msg)

    prev_end = None
    for ev in events:
        lines = (ev.get("text") or "").split("\n")
        _enforce(len(lines) <= 2, "more than 2 lines")
        if lines:
            _enforce(
                max(len(line) for line in lines) <= max_chars_per_line,
                "CPL > limit",
            )
        dur = ev["end"] - ev["start"]
        _enforce(dur + tolerance >= min_dur, "duration <20f")
        txt = "".join(lines)
        cps = len(txt) / max(0.001, dur)
        _enforce(cps <= cps_target + 1e-6, "cps > limit")
        if prev_end is not None:
            gap = ev["start"] - prev_end
            _enforce(gap >= min_gap_frames * spf - 1e-6, "gap <2f")
        prev_end = ev["end"]
    return events


def rebalance_cps_borrow_time(
    events: List[Dict[str, Any]],
    fps: float = 25.0,
    cps_target: float = 20.0,
    min_gap_frames: int = 2,
    min_dur_frames: int = 20,
) -> List[Dict[str, Any]]:
    spf = 1.0 / fps
    min_gap = min_gap_frames * spf
    min_dur = min_dur_frames * spf

    def chars_of(e: Dict[str, Any]) -> int:
        return len(" ".join((e.get("text") or "").split()))

    i = 0
    while i < len(events) - 1:
        e, nx = events[i], events[i + 1]
        dur = e["end"] - e["start"]
        chars = chars_of(e)
        need = max(0.0, chars / cps_target - dur)
        need = max(need, max(0.0, min_dur - dur))

        if need <= 1e-6:
            i += 1
            continue

        if i > 0 and need > 0:
            prev = events[i - 1]
            gap_left = e["start"] - prev["end"]
            left_slack = max(0.0, gap_left - min_gap)
            take_left = min(need, left_slack)
            if take_left > 0:
                e["start"] -= take_left
                need -= take_left

        if need <= 1e-6:
            i += 1
            continue

        gap = nx["start"] - e["end"]
        gap_slack = max(0.0, gap - min_gap)
        take_gap = min(need, gap_slack)
        if take_gap > 0:
            e["end"] += take_gap
            need -= take_gap

        if need > 0:
            nx_dur = nx["end"] - nx["start"]
            nx_chars = chars_of(nx)
            nx_min = max(min_dur, nx_chars / cps_target)
            room_nx = max(0.0, nx_dur - nx_min)

            give = min(need, room_nx)
            if give > 0:
                orig_e_end = e["end"]
                orig_nx_start = nx["start"]
                orig_nx_end = nx["end"]

                new_nx_start = nx["start"] + give
                ws = nx.get("words") or []
                nx_audio_end = ws[-1]["end"] if ws else orig_nx_end
                next_of_next_start = events[i + 2]["start"] if i + 2 < len(events) else float("inf")
                linger_cap = nx_audio_end + 0.5
                max_extend = min(linger_cap, next_of_next_start - min_gap) - orig_nx_end
                new_nx_end = orig_nx_end
                if max_extend > 0:
                    new_nx_end = min(orig_nx_end + give, orig_nx_end + max_extend)

                new_e_end = min(e["end"] + give, new_nx_start - min_gap)

                new_nx_dur = new_nx_end - new_nx_start
                new_nx_cps = chars_of(nx) / max(1e-3, new_nx_dur)

                if new_nx_dur + 1e-6 < min_dur or new_nx_cps > cps_target:
                    nx["start"] = orig_nx_start
                    nx["end"] = orig_nx_end
                    e["end"] = orig_e_end
                else:
                    nx["start"] = new_nx_start
                    nx["end"] = new_nx_end
                    e["end"] = new_e_end
                    need -= give
                    _dbg_add_ms(e, "borrow_from_right_ms", give * 1000.0)
                    _dbg_add_ms(nx, "gave_to_left_ms", give * 1000.0)

        i += 1
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
    min_readable: float = 1.20,
    min_two_line_chars: int = 24,
    max_block_duration_s: float = 7.0,
    max_merge_gap_ms: int = 360,
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
    # Fast-safe mode to prove where the stall is
    if os.environ.get("PARAKEET_TIMING_SAFE") == "1":
        _trace("SAFE mode: skipping packer, min_readable, and netflix normalizer")
        return events

    timed_out = False

    # Merge small neighbors into calm 2-line blocks (can be heavy)
    if os.environ.get("PARAKEET_DISABLE_PACKER") != "1":
        _trace(f"packer in: {len(events)}")
        _e = _with_timeout(5.0, pack_into_two_line_blocks,
                           events,
                           max_chars_per_line=max_chars_per_line,
                           cps_target=cps_target,
                           coalesce_gap_ms=coalesce_gap_ms,
                           two_line_threshold=two_line_threshold,
                           min_two_line_chars=min_two_line_chars,
                           max_block_duration_s=max_block_duration_s,
                           shaper=shape_words_into_two_lines_balanced)
        if _e is not None:
            events = _e
        else:
            timed_out = True
        _trace(f"packer out: {len(events)}")

    # Eliminate quick singles (orphans) and short flashes
    if os.environ.get("PARAKEET_DISABLE_MINREADABLE") != "1":
        _trace(f"min_readable in: {len(events)}")
        _e = _with_timeout(5.0, enforce_min_readable_v2,
                           events,
                           min_dur=min_readable,
                           cps_target=cps_target,
                           max_chars_per_line=max_chars_per_line,
                           min_two_line_chars=min_two_line_chars,
                           max_merge_gap_ms=max_merge_gap_ms,
                           shaper=shape_words_into_two_lines_balanced)
        if _e is not None:
            events = _e
        else:
            timed_out = True
        _trace(f"min_readable out: {len(events)}")

    # Netflix timing: linger-only-when-safe + chaining + 20f for 1–2 words
    if snap_fps and os.environ.get("PARAKEET_DISABLE_NETFLIX") != "1":
        _trace(f"netflix in: {len(events)}")
        _e = _with_timeout(5.0, normalize_timing_netflix,
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
                           shaper=shape_words_into_two_lines_balanced,
                           max_block_duration_s=max_block_duration_s,
                           validate=not timed_out)
        if _e is not None:
            events = _e
        else:
            timed_out = True
        _trace(f"netflix out: {len(events)}")

    if snap_fps and os.environ.get("PARAKEET_DISABLE_NETFLIX") != "1":
        events = rebalance_cps_borrow_time(
            events,
            fps=snap_fps,
            cps_target=cps_target,
            min_gap_frames=2,
            min_dur_frames=20,
        )
        _trace(f"netflix re-snap in: {len(events)}")
        _e = _with_timeout(5.0, normalize_timing_netflix,
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
                           shaper=shape_words_into_two_lines_balanced,
                           max_block_duration_s=max_block_duration_s,
                           validate=not timed_out)
        if _e is not None:
            events = _e
        else:
            timed_out = True
        _trace(f"netflix re-snap out: {len(events)}")
    if snap_fps and os.environ.get("PARAKEET_DISABLE_NETFLIX") != "1":
        _trace("final snap/validate in")
        _e = _with_timeout(3.0, normalize_timing_netflix,
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
                           shaper=shape_words_into_two_lines_balanced,
                           max_block_duration_s=max_block_duration_s,
                           validate=not timed_out)
        if _e is not None:
            events = _e
    def _infer_fps(events):
        # fallback if snap_fps is falsy; picks closest known grid
        cand = [23.976, 24.0, 25.0, 29.97, 30.0]
        def rmse(fps):
            spf = 1.0 / max(1e-9, fps)
            import math
            # measure how close starts/ends are to frame grid
            err = []
            for ev in events:
                for t in (ev["start"], ev["end"]):
                    q = round(t / spf)
                    err.append((t - q * spf) ** 2)
            return (sum(err) / max(1, len(err))) ** 0.5
        return min(cand, key=rmse)

    eff_fps = snap_fps or _infer_fps(events)
    spf = 1.0 / eff_fps
    for i in range(len(events) - 1):
        ns = events[i + 1]["start"]
        pe = events[i]["end"]
        min_gap = 2 * spf
        if ns - pe < min_gap - 1e-9:
            new_end = ns - min_gap
            if pe - new_end > 1e-9:
                _dbg_add_ms(events[i], "final_gap_fence_ms", (pe - new_end) * 1000.0)
            events[i]["end"] = max(new_end, events[i]["start"] + spf)
    return events