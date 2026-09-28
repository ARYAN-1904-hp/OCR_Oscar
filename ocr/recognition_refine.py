from __future__ import annotations

"""
Second-pass recognition fixes applied to PaddleOCR's word boxes.

1. refine_word_recognition()
   PaddleOCR's DB detector returns boxes cropped tightly to the x-height of
   the text. Glyphs that extend past it - parentheses above/below the
   digits, a comma's tail, the gap between two words - get clipped or
   squeezed, so the recognizer drops them: "(555) 881-9263" comes back as
   "555881-9263", "DAVIS, OLIVER" as "DAVIS,OLIVER", "11/17/2025 09:12" as
   "11/17/202509:12". Re-reading the same box with a little padding restores
   them. A re-read is only accepted when its letters/digits are *identical*
   (case-sensitive) to the first read and it differs only by restored
   spacing/brackets - padding also lets the recognizer hallucinate trailing
   glyphs from neighbouring borders, which this rejects.

2. recover_empty_field_values()
   The detector never proposes a box for a lone short value inside a form
   input (e.g. Status "2", POA Indicator "Y") - the box border dominates the
   crop and the tiny glyph scores below the box threshold. Input boxes are
   found from the image itself; any box with ink but no detected word gets
   its ink region recognized directly.
"""

import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# (vertical, horizontal) padding as a fraction of the box height. Two passes
# because neither alone recovers everything: 0.2 restores "PARTIAL APPROVAL",
# 0.35 restores "(555) 593-7414" and "PATEL MD, RAJ KUMAR".
_PADS: Tuple[Tuple[float, float], ...] = ((0.2, 0.1), (0.35, 0.2))

_MIN_ACCEPT_CONF = 0.85
_BRACKETS = "()[]"


def _alnum(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s)


def _balanced(s: str) -> bool:
    return s.count("(") == s.count(")") and s.count("[") == s.count("]")


def _edit_is_safe(old: str, new: str, old_conf: float, new_conf: float) -> bool:
    """True when `new` only restores spacing/punctuation missing from `old`."""
    if _alnum(old) != _alnum(new):
        return False
    if _balanced(old) and not _balanced(new):
        return False  # "555)593-7414": half a bracket pair is worse than none
    if new_conf < _MIN_ACCEPT_CONF:
        return False

    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        a_seg, b_seg = old[i1:i2], new[j1:j2]
        if op == "insert":
            for k, ch in enumerate(b_seg):
                pos = j1 + k
                if ch == " ":
                    # A space between two lowercase letters splits a word
                    # ("Professio nal"); real restored gaps sit next to
                    # punctuation, digits, or a case boundary.
                    left = new[pos - 1] if pos > 0 else ""
                    right = new[pos + 1] if pos + 1 < len(new) else ""
                    if left.islower() and right.islower():
                        return False
                elif ch in _BRACKETS:
                    continue
                elif ch == ":" and pos == len(new) - 1:
                    continue  # "#2(23)" -> "#2(23):"
                else:
                    return False  # e.g. trailing "." read off a box border
        elif op == "delete":
            # Only a spurious space inside an all-caps word ("RESPIRAT ORY").
            if a_seg != " ":
                return False
            left = old[i1 - 1] if i1 > 0 else ""
            right = old[i2] if i2 < len(old) else ""
            if not (left.isupper() and right.isupper()):
                return False
        elif op == "replace":
            # Punctuation swap ("$2.844.42" -> "$2,844.42"); the two reads
            # disagree about which is right, so require a clearly better read.
            if any(c.isalnum() for c in a_seg + b_seg):
                return False
            if new_conf < old_conf + 0.03:
                return False
    return True


def _crop(img: np.ndarray, bbox: List[int], pad_v: float, pad_h: float) -> Optional[np.ndarray]:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    h = max(1, y2 - y1)
    py = max(2, int(h * pad_v))
    px = max(2, int(h * pad_h))
    c = img[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]
    if c.size == 0 or c.shape[0] < 4 or c.shape[1] < 4:
        return None
    # The recognizer was validated on BGR crops (cv2 convention).
    return np.ascontiguousarray(c[:, :, ::-1]) if c.ndim == 3 else c


def _recognize(ocr: Any, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
    rec = getattr(ocr, "text_recognizer", None)
    if rec is None or not crops:
        return []
    try:
        res, _ = rec(crops)
    except Exception as e:  # never let a refinement pass fail the page
        print(f"[recognition_refine] re-recognition failed: {e}")
        return []
    return [(str(t or ""), float(c or 0.0)) for t, c in res]


_ENGLISH_WORDS: Optional[set] = None


def _english_words() -> set:
    """Lowercase English vocabulary (symspellpy's bundled frequency list)."""
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        vocab: set = set()
        try:
            from importlib import resources

            path = resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
            with path.open(encoding="utf-8") as fh:
                vocab = {line.split(" ", 1)[0] for line in fh if line.strip()}
        except Exception:
            pass
        _ENGLISH_WORDS = vocab
    return _ENGLISH_WORDS


_CAPS_RUN_RE = re.compile(r"[A-Z]{7,}")
_SUSPECT_RES = [
    re.compile(r"\d{6,}-\d"),                   # "555881-9263": area-code parens lost
    re.compile(r"^\d{3}\)|\d\)\d"),             # "555)593-7414": half a pair read
    re.compile(r"\$\d{1,3}\.\d{3}[.,]\d"),      # "$2.844.42": thousands comma read as "."
    re.compile(r"\(\d+\)$"),                    # "#2(23)": trailing colon lost
    re.compile(r"[a-z]/[A-Z]"),                 # "Coverage/Eligibility": spaces lost
    re.compile(r"\b[A-Z]\d[A-Z]{2,}\b"),        # "P3CA": space lost
]


def _needs_reread(text: str) -> bool:
    """Cheap filter for reads that look clipped, so not every box pays for a re-read.

    Recognition is the expensive step on CPU (~0.5s/box single-threaded), and
    spacing after , : & / and date-time jams are already repaired in text by
    ocr.postprocess - only losses a regex can't undo are worth re-reading.
    """
    t = (text or "").strip()
    if not t:
        return False
    if not _balanced(t):
        return True
    if any(p.search(t) for p in _SUSPECT_RES):
        return True
    vocab = _english_words()
    for run in _CAPS_RUN_RE.findall(t):
        # Two capitalised words fused together ("PARTIALAPPROVAL", "PATELMD",
        # "LOWBACK") or split mid-word ("RESPIRAT ORY") aren't dictionary words.
        if not vocab or run.lower() not in vocab:
            return True
    return False


def refine_word_recognition(ocr: Any, img_rgb: np.ndarray, words: List[Dict]) -> List[Dict]:
    """Re-read suspicious word boxes with padding; keep reads that restore punctuation/spacing."""
    if not words or img_rgb is None:
        return words

    pending = [i for i, w in enumerate(words) if _needs_reread(w.get("text") or "")]
    best: Dict[int, Tuple[str, float]] = {}
    for pad_v, pad_h in _PADS:
        idx: List[int] = []
        crops: List[np.ndarray] = []
        for i in pending:
            c = _crop(img_rgb, words[i]["bbox"], pad_v, pad_h)
            if c is not None:
                idx.append(i)
                crops.append(c)
        results = _recognize(ocr, crops)
        if len(results) != len(crops):
            break
        for i, (text, conf) in zip(idx, results):
            w = words[i]
            old = (w.get("text") or "").strip()
            new = text.strip()
            if not new or new == old:
                continue
            if not _edit_is_safe(old, new, float(w.get("conf", 0.0)), conf):
                continue
            if i not in best or conf > best[i][1]:
                best[i] = (new, conf)
        # Only boxes the first padding couldn't fix get the second try.
        pending = [i for i in pending if i not in best]
        if not pending:
            break

    out = []
    for i, w in enumerate(words):
        if i in best:
            w = dict(w)
            w["text_raw"] = w.get("text")
            w["text"], w["conf"] = best[i]
        out.append(w)
    return out


# UI icon glyphs (checkbox squares, close crosses, search circles) come back
# as 1-3 low-confidence O/0/X characters: "Professional Claim ☐ ×" read as
# "Professional Claim 0x", the toolbar's ✕ button as "X".
_ICON_GLYPH_RE = re.compile(r"^[0OoXx×\s]{1,3}$")
_ICON_MAX_CONF = 0.80


def drop_icon_glyphs(words: List[Dict]) -> List[Dict]:
    return [
        w for w in words
        if not (
            _ICON_GLYPH_RE.match((w.get("text") or "").strip() or "?")
            and float(w.get("conf", 1.0)) < _ICON_MAX_CONF
        )
    ]


# -- Missing values inside form input boxes --------------------------------

def _has_border(gray: np.ndarray, box: List[int]) -> bool:
    """True if all four sides of `box` are drawn lines, not the edge of a text blob.

    Canny also closes contours around bold words/words runs; those have ink
    only where glyphs are, so their "edges" fail this continuity check.
    """
    x1, y1, x2, y2 = box
    H, W = gray.shape[:2]
    if x2 - x1 < 16 or y2 - y1 < 16 or x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return False
    g = gray.astype(np.int16)
    # The contour's bounding rect sits up to ~2px outside the drawn line
    # (Canny + dilation), so look for the line within a 4px band on each
    # side and compare it with the background 6px in.
    b, d, m = 4, 6, 6
    sides = [
        (g[y1:y1 + b, x1 + m:x2 - m].min(axis=0), g[y1 + d, x1 + m:x2 - m]),
        (g[y2 - b:y2, x1 + m:x2 - m].min(axis=0), g[y2 - 1 - d, x1 + m:x2 - m]),
        (g[y1 + m:y2 - m, x1:x1 + b].min(axis=1), g[y1 + m:y2 - m, x1 + d]),
        (g[y1 + m:y2 - m, x2 - b:x2].min(axis=1), g[y1 + m:y2 - m, x2 - 1 - d]),
    ]
    for line, inside in sides:
        if line.size == 0:
            return False
        if float(np.mean(inside - line >= 12)) < 0.85:
            return False
    return True


def _find_input_boxes(img_rgb: np.ndarray) -> List[List[int]]:
    """Axis-aligned rectangular outlines shaped like form text inputs."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape[:2]
    edges = cv2.Canny(gray, 20, 60)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[List[int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (14 <= h <= 60 and w >= 2.5 * h and w <= W * 0.6):
            continue
        # Outline must roughly fill its bounding rectangle (a real box, not text).
        if cv2.contourArea(cnt) < 0.6 * w * h:
            continue
        if not _has_border(gray, [x, y, x + w, y + h]):
            continue
        boxes.append([x, y, x + w, y + h])

    # Inner/outer edge of the same border both show up; keep the outer one.
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    kept: List[List[int]] = []
    for b in boxes:
        if any(b[0] >= k[0] - 3 and b[1] >= k[1] - 3 and b[2] <= k[2] + 3 and b[3] <= k[3] + 3 for k in kept):
            continue
        kept.append(b)
    return kept


def _overlaps(a: List[int], b: List[int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def recover_empty_field_values(ocr: Any, img_rgb: np.ndarray, words: List[Dict]) -> List[Dict]:
    """Recognize text inside input boxes that the detector returned no word for."""
    if img_rgb is None or img_rgb.ndim != 3:
        return words
    try:
        boxes = _find_input_boxes(img_rgb)
    except Exception as e:
        print(f"[recognition_refine] input box detection failed: {e}")
        return words

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape[:2]
    targets: List[List[int]] = []
    crops: List[np.ndarray] = []
    for box in boxes:
        # Any detected word touching the box means the detector already saw
        # this field's content (or the "box" is really part of a text run).
        if any(_overlaps(w["bbox"], box) for w in words):
            continue
        bx1, by1, bx2, by2 = box
        inset = 3
        inner = gray[by1 + inset:by2 - inset, bx1 + inset:bx2 - inset]
        if inner.size == 0:
            continue
        ink = inner < 140
        if int(ink.sum()) < 8:
            continue  # genuinely empty field
        if float(ink.mean()) > 0.25:
            continue  # a filled badge/button, not a text input
        ys, xs = np.nonzero(ink)
        ix1, ix2 = bx1 + inset + int(xs.min()), bx1 + inset + int(xs.max()) + 1
        iy1, iy2 = by1 + inset + int(ys.min()), by1 + inset + int(ys.max()) + 1
        bh = by2 - by1
        # Input text is left-aligned just inside the border; a glyph centred
        # in its box is a toolbar/icon button, not a field value.
        if ix1 - bx1 > max(12, 0.7 * bh):
            continue
        pad = max(3, (iy2 - iy1) // 3)
        c = img_rgb[max(0, iy1 - pad):min(H, iy2 + pad), max(0, ix1 - pad):min(W, ix2 + pad)]
        if c.size == 0:
            continue
        targets.append([ix1, iy1, ix2, iy2])
        crops.append(np.ascontiguousarray(c[:, :, ::-1]))

    results = _recognize(ocr, crops)
    if len(results) != len(crops):
        return words

    out = list(words)
    for bbox, (text, conf) in zip(targets, results):
        t = text.strip()
        if not t or conf < 0.5 or not any(ch.isalnum() for ch in t):
            continue
        if _ICON_GLYPH_RE.match(t) and conf < _ICON_MAX_CONF:
            continue  # e.g. a search box's magnifier read as "0"
        out.append({"text": t, "conf": conf, "bbox": bbox, "source": "input_box_probe"})
    return out
