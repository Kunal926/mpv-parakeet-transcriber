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


def _fits_line(cur_words, new_word_text, max_chars):
    t = " ".join(cur_words + [new_word_text])
    return len(t) <= max_chars


def shape_words_into_two_lines_no_loss(words, max_chars, max_lines=2):
    """
    Input: list[{"word","start","end"}]
    Returns: (lines_text:list[str], used_words:int, overflow_words:list[dict])
    - uses linguistic hints (punctuation, conj/prep) to place the break
    - preserves words => we can set exact start/end times
    """
    if not words:
        return [""], 0, []
    line1, line2 = [], []
    used = 0

    # Greedy fill line1
    for i, w in enumerate(words):
        wt = (w.get("word") or "").strip()
        if not _fits_line([x for x in (line1)], wt, max_chars):
            break
        line1.append(wt); used = i + 1
        # prefer break after punctuation if close to limit
        if wt.endswith((".", "!", "?", "…", ",", ":", ";")) and len(" ".join(line1)) >= max_chars * 0.7:
            break

    rem = words[used:]

    # If nothing left, single-line
    if not rem or max_lines == 1:
        return [" ".join(line1)], used, []

    # Fill line2 up to max_chars (no trimming)
    for j, w in enumerate(rem):
        wt = (w.get("word") or "").strip()
        if not _fits_line([x for x in (line2)], wt, max_chars):
            overflow_words = rem[j:]
            return [" ".join(line1), " ".join(line2)], used + j, overflow_words
        line2.append(wt)

    return [" ".join(line1), " ".join(line2)], len(words), []

def segment_by_pause_and_phrase(words: List[Dict[str,Any]],
                                max_chars_per_line: int = 46,
                                max_lines: int = 2,
                                pause_ms: int = 220,
                                cps_target: float = 20.0,
                                use_spacy: bool = True,
                                spacy_model: str = "en_core_web_sm") -> List[Dict[str,Any]]:
    """
    Build segments from ASR word-level tokens using:
      - hard breaks at end punctuation
      - soft breaks at commas or conjunctions/prepositions + pause
      - breaks at pauses (word gap >= pause_ms)
    Keep zero-loss (all words appear). Return list of dicts {start,end,text,words}.
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

        # primary break rule (ranked)
        should_break = False
        reason = ""
        if ends_hard:
            should_break, reason = True, "endpunct"
        elif gap >= pause_ms and (ends_soft or conj_or_prep_next):
            should_break, reason = True, "pause+soft"
        elif gap >= pause_ms * 1.33:
            should_break, reason = True, "pause"
        else:
            # reading speed pressure: if buffer is dense and there is a soft boundary, cut here
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

    # two-line shaping that PRESERVES word timings
    shaped = []
    for seg in out:
        cur_words = seg.get("words") or []
        if not cur_words:
            # fallback: keep text; no overflow splitting
            shaped.append(seg)
            continue

        start_idx = 0
        words_list = cur_words
        while words_list:
            lines_text, used_words, overflow = shape_words_into_two_lines_no_loss(
                words_list, max_chars_per_line, max_lines
            )

            used = words_list[:used_words]
            new_seg = {
                "start": float(used[0]["start"]),
                "end":   float(used[-1]["end"]),
                "text":  "\n".join(lines_text),
                "words": used,
            }
            shaped.append(new_seg)

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
