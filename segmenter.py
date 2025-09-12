from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import logging, re, math

PUNCT_END = (".", "!", "?", "…", ":", ";")
SOFT_PUNCT_END = (",",)  # may end a phrase if there is also a pause
CONJ = {"and","but","or","nor","so","yet","for","because","although","though","if","when","while"}
PREP = {"about","above","across","after","against","along","among","around","at","before","behind",
        "below","beneath","beside","between","beyond","by","despite","down","during","except","for",
        "from","in","inside","into","like","near","of","off","on","onto","out","outside","over","past",
        "since","through","throughout","to","toward","under","underneath","until","up","upon","with",
        "within","without"}


def _normalize(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _token_text(w: Dict[str, Any]) -> str:
    # Many ASR word tokens include trailing punctuation, keep it for phrase tests.
    return (w.get("word") or "").strip()


def _gap_ms(a_end: float, b_start: float) -> float:
    return max(0.0, (b_start - a_end) * 1000.0)


def _is_titlecase_phrase(words: List[Dict[str,Any]], i: int, j: int) -> bool:
    # crude “do not break inside” for Names / Proper Noun sequences
    toks = [_token_text(w) for w in words[i:j]]
    cap = sum(1 for t in toks if t[:1].isupper())
    return cap >= max(2, (j - i))


def _load_spacy(model: str="en_core_web_sm"):
    try:
        import spacy
        return spacy.load(model, disable=["lemma","ner"])
    except Exception as e:
        logging.info("spaCy unavailable (%s) — falling back to rule-only.", e)
        return None


def _spacy_bad_boundary(nlp, text: str, left_text: str, right_text: str) -> bool:
    """
    If spaCy is present, discourage breaking between tokens that belong together:
    - determiners/adjectives + noun
    - auxiliaries/negations + verb
    - preposition + its object
    """
    if not nlp: return False
    doc = nlp(_normalize(left_text + " | " + right_text))
    # find the '|' boundary token index
    bar = [i for i,t in enumerate(doc) if t.text == "|"]
    if not bar: return False
    k = bar[0]
    # left token (k-1) and right token (k+1)
    L = doc[k-1] if k-1>=0 else None
    R = doc[k+1] if k+1 < len(doc) else None
    if not L or not R: return False
    # keep DET/ADJ with NOUN; keep PART/ADV/NEG with VERB/AUX; keep PREP with pobj
    bad = (
        (L.pos_ in {"DET","ADJ"} and R.pos_ in {"NOUN","PROPN"}) or
        (L.dep_ in {"neg"} and R.pos_ in {"VERB","AUX"}) or
        (L.pos_ in {"AUX"} and R.pos_ in {"VERB"}) or
        (L.pos_ == "ADP" and R.dep_ == "pobj")
    )
    return bool(bad)


def _words_text(words):
    return " ".join((w.get("word") or "").strip() for w in words).strip()


def shape_words_into_two_lines_balanced(
    words,
    max_chars,
    prefer_two_lines: bool = True,
    two_line_threshold: float = 0.72,
):
    """
    Input: list of word dicts with 'word','start','end'
    Returns: (lines_text:list[str], used_words:int, overflow_words:list[dict])
    - Picks the split that best balances line lengths while respecting soft rules.
    - Never drops words; overflow becomes continuation.
    """
    toks = [(w.get("word") or "").strip() for w in words]
    if not toks:
        return [""], 0, []

    total = " ".join(toks)
    # If it comfortably fits in one line and we don't prefer two lines, keep it.
    if not prefer_two_lines and len(total) <= max_chars:
        return [total], len(words), []

    # If we do prefer two lines and it's “long enough”, try to split anyway.
    want_two = prefer_two_lines and (len(total) >= int(two_line_threshold * max_chars))

    # Candidate breakpoints between words 1..n-1
    best, best_score = None, -1e9
    for cut in range(1, len(toks)):
        left = " ".join(toks[:cut]); right = " ".join(toks[cut:])
        # hard limits (no trimming)
        if len(left) > max_chars or len(right) > max_chars:
            continue

        # heuristics
        prev = toks[cut-1].strip(",.;:!?…").lower()
        cur  = toks[cut].strip(",.;:!?…").lower()

        score = 0.0
        # balance: minimize absolute difference
        score -= abs(len(left) - len(right)) * 1.2
        # prefer after punctuation / before conj or prep
        if toks[cut-1][-1:] in (".","!","?","…",",",":",";"): score += 3.0
        if cur in {"and","but","or","nor","so","yet","for","because","although","though","if","when","while"}: score += 2.0
        if cur in {"about","above","across","after","against","along","among","around","at","before","behind","below","beneath","beside","between","beyond","by","despite","down","during","except","for","from","in","inside","into","like","near","of","off","on","onto","out","outside","over","past","since","through","throughout","to","toward","under","underneath","until","up","upon","with","within","without"}: score += 1.5

        # avoid splitting tight pairs
        if prev in {"a","an","the"}: score -= 6.0
        if prev in {"i","you","he","she","we","they","it"}: score -= 2.0
        # avoid very short lines
        if len(left) < 8 or len(right) < 8: score -= 3.0

        if score > best_score:
            best_score, best = score, cut

    if best is None:
        # no legal split
        if len(total) <= max_chars or not want_two:
            return [total], len(words), []
        # force a safe split at the largest fitting point
        cut = max(i for i in range(1, len(toks)) if len(" ".join(toks[:i])) <= max_chars)
        best = cut

    left = " ".join(toks[:best]); right = " ".join(toks[best:])
    if len(right) > max_chars:
        # overflow flows to continuation (no trimming)
        # cut right to max_chars without splitting a word; rest is overflow
        cut_text = right[:max_chars+1]
        if " " in cut_text:
            cut_idx = cut_text.rfind(" ")
            right_vis = right[:cut_idx]
            overflow  = right[cut_idx+1:].strip().split(" ")
            used_words = best + len(right_vis.split(" "))
            return [left, right_vis], used_words, words[used_words:]
        else:
            # pathological long token; keep one line
            return [left, right], best + len(right.split(" ")), []
    used_words = best + len(right.split(" "))
    return [left, right], used_words, words[used_words:]


def coalesce_short_neighbors(
    events,
    gap_ms: int = 180,
    max_chars_per_line: int = 42,
    cps_target: float = 20.0,
):
    i = 0
    while i + 1 < len(events):
        a, b = events[i], events[i + 1]
        gap = (b["start"] - a["end"]) * 1000.0
        dur = (b["end"] - a["start"])
        chars = len(a["text"].replace("\n", " ")) + 1 + len(b["text"].replace("\n", " "))
        cps = chars / max(0.001, dur)
        if gap <= gap_ms and cps <= cps_target and chars <= max_chars_per_line * 2:
            # merge into one event; keep words so timing stays exact
            a["text"] = a["text"].rstrip() + "\n" + b["text"].lstrip()
            a["end"] = b["end"]
            if a.get("words") and b.get("words"):
                a["words"] = a["words"] + b["words"]
            del events[i + 1]
        else:
            i += 1
    return events

def segment_by_pause_and_phrase(
    words: List[Dict[str, Any]],
    max_chars_per_line: int = 46,
    max_lines: int = 2,
    pause_ms: int = 220,
    punct_pause_ms: int = 160,
    comma_pause_ms: int = 120,
    cps_target: float = 20.0,
    use_spacy: bool = True,
    spacy_model: str = "en_core_web_sm",
    two_line_threshold: float = 0.70,
) -> List[Dict[str, Any]]:
    """
    Build segments from ASR word-level tokens using pauses as the primary cue.
    Punctuation only triggers a split when paired with a sufficient pause.
    Keeps zero-loss (all words appear). Return list of dicts {start,end,text,words}.
    """
    nlp = _load_spacy(spacy_model) if use_spacy else None
    out, buf, buf_start = [], [], None

    def flush():
        nonlocal buf, buf_start
        if not buf: return
        seg = {
            "start": float(buf_start if buf_start is not None else buf[0]["start"]),
            "end": float(buf[-1]["end"]),
            "text": _normalize(" ".join(_token_text(w) for w in buf)),
            "words": buf[:],
        }
        out.append(seg)
        buf, buf_start = [], None

    for i, w in enumerate(words):
        t = _token_text(w)
        if not buf:
            buf_start = w["start"]
        buf.append(w)
        # lookahead
        nxt = words[i+1] if i+1 < len(words) else None
        gap = _gap_ms(w["end"], nxt["start"]) if nxt else 0.0

        # natural phrase endings
        ends_hard = t.endswith(PUNCT_END)
        ends_soft = t.endswith(SOFT_PUNCT_END)

        # conjunction/preposition boundary (prefer BEFORE conj/prep)
        conj_or_prep_next = False
        if nxt:
            ntok = _token_text(nxt).lower().strip("“”\"'()[]")
            if ntok in CONJ or ntok in PREP:
                conj_or_prep_next = True

        # Break primarily on PAUSE; punctuation only strengthens the decision.
        should_break = False
        reason = ""
        if gap >= pause_ms:
            # long enough silence: OK to break anywhere
            should_break, reason = True, "pause"
        elif ends_hard and gap >= punct_pause_ms:
            # sentence-ending punctuation needs at least a small pause
            should_break, reason = True, "punct+pause"
        elif ends_soft and gap >= comma_pause_ms:
            # comma needs a pause too, usually shorter
            should_break, reason = True, "comma+pause"
        else:
            # reading speed pressure: if dense *and* we’re at a soft boundary,
            # allow a break to keep cps comfy
            dur = (buf[-1]["end"] - buf[0]["start"])
            chars = len(_normalize(" ".join(_token_text(x) for x in buf)))
            cps = chars / max(0.001, dur)
            if cps > cps_target and (ends_soft or conj_or_prep_next):
                should_break, reason = True, "cps"

        # spaCy veto: avoid bad breaks inside tight word groups
        if should_break and nlp and nxt:
            left_text  = " ".join(_token_text(x) for x in buf)
            right_text = _token_text(nxt)
            if _spacy_bad_boundary(nlp, left_text, left_text, right_text):
                should_break = False

        # Title‑case phrase protection (e.g., "Sana's Village", "Hydra Attack")
        if should_break and _is_titlecase_phrase(words, max(0, i-1), i+1):
            should_break = False

        if should_break:
            flush()

    flush()

    out = coalesce_short_neighbors(
        out,
        gap_ms=180,
        max_chars_per_line=max_chars_per_line,
        cps_target=cps_target,
    )

    # two-line shaping (no-loss, word-timing preserving)
    shaped = []
    for seg in out:
        cur_words = seg.get("words") or []
        if not cur_words:
            shaped.append(seg)
            continue

        words_list = cur_words
        while words_list:
            lines_text, used_words, overflow = shape_words_into_two_lines_balanced(
                words_list,
                max_chars_per_line,
                prefer_two_lines=True,
                two_line_threshold=two_line_threshold,
            )

            used_block = words_list[:used_words]
            new_seg = {
                "start": float(used_block[0]["start"]),
                "end": float(used_block[-1]["end"]),
                "text": "\n".join(lines_text[:2]),
                "words": used_block,
            }
            shaped.append(new_seg)

            # Next iteration processes the true overflow words
            words_list = overflow
    return shaped


def shape_lines_no_loss(text: str, max_chars: int, max_lines: int) -> Tuple[list[str], str]:
    text = _normalize(text)
    if max_lines <= 1 or len(text) <= max_chars:
        return [text], ""
    words = text.split(" ")
    # greedy fill for line1
    line1, line2 = [], []
    for w in words:
        trial = " ".join(line1 + [w])
        if len(trial) <= max_chars:
            line1.append(w); continue
        break
    rem = words[len(line1):]
    # prefer breaks after punctuation / before conj/prep
    if rem and line1:
        if not (line1[-1][-1:] in PUNCT_END or line1[-1][-1:] in SOFT_PUNCT_END):
            # try to move last word(s) to line2 if nicer
            while line1 and (len(" ".join(line1)) > max_chars or
                             (rem and (rem[0].lower() in CONJ or rem[0].lower() in PREP))):
                rem.insert(0, line1.pop())
    l1 = " ".join(line1).strip()
    l2 = " ".join(rem).strip()
    if len(l2) > max_chars:
        # overflow becomes continuation; do not trim visible text
        cut = l2[:max_chars+1].rsplit(" ", 1)[0] if " " in l2[:max_chars+1] else l2[:max_chars]
        overflow = l2[len(cut):].strip()
        return [l1, cut], overflow
    return [l1, l2], ""
