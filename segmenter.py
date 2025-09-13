from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import logging, re

# ---------- lexicon / heuristics ----------
CONJ = {"and","but","or","nor","so","yet","for","because","although","though","if","when","while"}
PREP = {"about","above","across","after","against","along","among","around","at","before","behind","below","beneath","beside",
        "between","beyond","by","despite","down","during","except","for","from","in","inside","into","like","near","of","off",
        "on","onto","out","outside","over","past","since","through","throughout","to","toward","under","underneath","until",
        "up","upon","with","within","without"}
HARD_PUNCT = (".","!","?","…",":",";")
SOFT_PUNCT = (",",)

def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def _wtext(w: Dict[str,Any]) -> str:
    return (w.get("word") or "").strip()

def _load_spacy(model="en_core_web_sm"):
    try:
        import spacy
        return spacy.load(model, disable=["lemma","ner"])
    except Exception as e:
        logging.info("spaCy unavailable: %s", e)
        return None

def _spacy_bad_boundary(nlp, left_text: str, right_text: str) -> bool:
    if not nlp: return False
    doc = nlp(_norm(left_text + " | " + right_text))
    bars = [i for i,t in enumerate(doc) if t.text == "|"]
    if not bars: return False
    k = bars[0]
    L = doc[k-1] if k-1>=0 else None
    R = doc[k+1] if k+1<len(doc) else None
    if not L or not R: return False
    # keep DET/ADJ with NOUN/PROPN; keep NEG/AUX with VERB; keep ADP with pobj
    bad = (
        (L.pos_ in {"DET","ADJ"} and R.pos_ in {"NOUN","PROPN"}) or
        (L.dep_ in {"neg"} and R.pos_ in {"VERB","AUX"}) or
        (L.pos_ in {"AUX"} and R.pos_ in {"VERB"}) or
        (L.pos_ == "ADP" and R.dep_ == "pobj")
    )
    return bool(bad)

# ---------- bottom-heavy 2-line shaping (word-preserving) ----------
def shape_words_into_two_lines_balanced(
    words: List[Dict[str, Any]],
    max_chars: int,
    prefer_two_lines: bool = True,
    two_line_threshold: float = 0.70,
    min_two_line_chars: int = 24,
) -> Tuple[List[str], int, List[Dict[str, Any]]]:
    toks = [_wtext(w) for w in words]
    if not toks:
        return [""], 0, []
    total = " ".join(toks)
    want_two = prefer_two_lines and (
        len(total) >= max(int(two_line_threshold*max_chars), min_two_line_chars)
    )
    # Try all boundaries between words
    best_cut, best_score = None, -1e9
    for cut in range(1, len(toks)):
        left = " ".join(toks[:cut]); right = " ".join(toks[cut:])
        if len(left) > max_chars or len(right) > max_chars:
            continue
        prev = toks[cut-1].strip(",.;:!?…").lower()
        cur  = toks[cut].strip(",.;:!?…").lower()
        L, R = len(left), len(right)
        score = 0.0
        # balance with bottom-heavy preference
        score -= abs(L - R) * 0.8
        if R >= L: score += 1.0
        else:      score -= 1.0
        # avoid 1–2 words on top or bottom line
        if len(left.split()) <= 2: score -= 6.0
        if len(right.split()) <= 2: score -= 6.0
        # linguistic: after punctuation / before conj/prep
        if toks[cut-1][-1:] in HARD_PUNCT + SOFT_PUNCT: score += 3.0
        if cur in CONJ: score += 2.0
        if cur in PREP: score += 1.5
        # avoid breaking after article / subj pronoun
        if prev in {"a","an","the"}: score -= 6.0
        if prev in {"i","you","he","she","we","they","it"}: score -= 2.0
        # avoid very short lines
        if L < 8 or R < 8: score -= 3.0
        if score > best_score:
            best_score, best_cut = score, cut
    if best_cut is None:
        if not want_two or len(total) <= max_chars:
            return [total], len(words), []
        # force largest fitting cut
        cut = max(i for i in range(1, len(toks)) if len(" ".join(toks[:i])) <= max_chars)
        best_cut = cut
    left = " ".join(toks[:best_cut]); right = " ".join(toks[best_cut:])
    if len(right) > max_chars:
        # split right side further; overflow becomes continuation words
        vis = right[:max_chars+1]
        if " " in vis:
            k = vis.rfind(" ")
            right_vis = right[:k]
            used = best_cut + len(right_vis.split())
            return [left, right_vis], used, words[used:]
        return [left, right], best_cut + len(right.split()), []
    used = best_cut + len(right.split())
    return [left, right], used, words[used:]

# ---------- pause+phrase segmentation ----------
def segment_by_pause_and_phrase(
    words: List[Dict[str,Any]],
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    pause_ms: int = 240,
    punct_pause_ms: int = 160,
    comma_pause_ms: int = 120,
    cps_target: float = 20.0,
    use_spacy: bool = True,
    spacy_model: str = "en_core_web_sm",
    two_line_threshold: float = 0.70,
    min_two_line_chars: int = 24,
) -> List[Dict[str,Any]]:
    nlp = _load_spacy(spacy_model) if use_spacy else None
    out, buf, buf_start = [], [], None
    def flush():
        nonlocal buf, buf_start
        if not buf: return
        seg = {
            "start": float(buf[0]["start"]),
            "end":   float(buf[-1]["end"]),
            "text":  _norm(" ".join(_wtext(w) for w in buf)),
            "words": buf[:],
        }
        out.append(seg); buf, buf_start = [], None
    for i, w in enumerate(words):
        t = _wtext(w)
        if not buf: buf_start = w["start"]
        buf.append(w)
        nxt = words[i+1] if i+1<len(words) else None
        gap = (nxt["start"] - w["end"])*1000.0 if nxt else 0.0
        ends_hard = t.endswith(HARD_PUNCT)
        ends_soft = t.endswith(SOFT_PUNCT)
        conj_or_prep_next = False
        if nxt:
            nt = _wtext(nxt).lower().strip("“”\"'()[]")
            if nt in CONJ or nt in PREP: conj_or_prep_next = True
        should_break = False
        # Primary: break on pause; punctuation only helps if there is a pause
        if gap >= pause_ms:
            should_break = True
        elif ends_hard and gap >= punct_pause_ms:
            should_break = True
        elif ends_soft and gap >= comma_pause_ms:
            should_break = True
        else:
            # cps pressure at a soft boundary
            dur = buf[-1]["end"] - buf[0]["start"]
            chars = len(_norm(" ".join(_wtext(x) for x in buf)))
            cps = chars / max(0.001, dur)
            if cps > cps_target and (ends_soft or conj_or_prep_next):
                should_break = True
        # spaCy veto for tight groups
        if should_break and nlp and nxt:
            left_text = " ".join(_wtext(x) for x in buf)
            right_text = _wtext(nxt)
            if _spacy_bad_boundary(nlp, left_text, right_text):
                should_break = False
        if should_break:
            flush()
    flush()
    # 2-line shaping preserving word timings, allowing continuations
    shaped: List[Dict[str, Any]] = []
    for seg in out:
        words_list = seg.get("words") or []
        if not words_list:
            shaped.append(seg)
            continue
        while words_list:
            lines, used, overflow = shape_words_into_two_lines_balanced(
                words_list,
                max_chars=max_chars_per_line,
                prefer_two_lines=True,
                two_line_threshold=two_line_threshold,
                min_two_line_chars=min_two_line_chars,
            )
            used_block = words_list[:used]
            shaped.append(
                {
                    "start": float(used_block[0]["start"]),
                    "end": float(used_block[-1]["end"]),
                    "text": "\n".join(lines[:2]),
                    "words": used_block,
                }
            )
            words_list = overflow
    return shaped
