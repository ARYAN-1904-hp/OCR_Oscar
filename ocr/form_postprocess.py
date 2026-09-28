"""
Post-process OCR pages for form accuracy:
  • Checkbox / Yes-No state detection (image ink + unicode + cell text)
  • Two-column form table reconstruction from word geometry
  • Cleanup of messy PPStructure tables into label/value grids
"""

from __future__ import annotations

import re
import statistics
from html import escape
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ocr.schemas import PageResult

_CHECK_MARKS_CHECKED = set("☒☑✓✔■▪●◉▣")
_CHECK_MARKS_UNCHECKED = set("☐□❍○◻▢")

# A bare "<...> Indicator" token with nothing else on it (no colon, no
# attached value - unlike e.g. "POA Indicator: YES, present on admission",
# which already carries its own value and isn't a checkbox) is this claims
# UI's convention for a standalone Yes/No-style checkbox field (EOB
# Indicator, MPPR Indicator, ...). Gating on this exact shape keeps the
# pixel probe below from ever running against ordinary headings/labels.
_STANDALONE_INDICATOR_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9 /&()#.'-]{1,50}\bIndicator$"
)

# Fallback vocabulary only - used when a cell has no colon AND no word-geometry
# gap to split on (e.g. a single jammed OCR token like "Suite#101"). The
# primary splitter (_split_label_value_geom / the generic colon regex below)
# works on any label it has never seen, so this list should stay a last
# resort rather than the thing new forms need to be added to.
_FORM_LABELS = sorted(
    [
        "Translation Services Available",
        "Business Hours Schedule",
        "Federal Tax ID (if different from above)",
        "Federal Tax ID",
        "Tax ID",
        "Provider Information",
        "Provider NPI",
        "Provider Name",
        "Specialty Code",
        "Specialty",
        "Service Address",
        "Billing Address",
        "Contract Type",
        "Contract",
        "Network Tier",
        "Network",
        "Referring Provider",
        "Provider Status",
        "Effective Date",
        "Term Date",
        "Decision SLA",
        "Processor ID",
        "Processor",
        "Last edit",
        "NPI # (if different than above)",
        "NPI # (if different from above)",
        "NPI # (lf different than above)",
        "Primary Facility Location",
        "Second Facility Location (if applicable)",
        "Legal Business Name",
        "Street Address",
        "Suite #",
        "Suite#",
        "Zip Code",
        "TTY Number",
        "ITY Number",
        "County",
        "Phone",
        "Fax",
        "Email",
        "City",
        "State",
        "Zip",
        "NPI",
    ],
    key=len,
    reverse=True,
)

_LABEL_SPLIT_RE = re.compile(
    r"^(?P<label>" + "|".join(re.escape(l) for l in _FORM_LABELS) + r")\s*[:.#]?\s*(?P<value>.*)$",
    re.I,
)


class _TableGridExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: List[str] = []
        self._cell: List[str] = []
        self._in_cell = False
        self._colspan = 1

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs or [])
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._in_cell = True
            try:
                self._colspan = max(1, int(attrs_d.get("colspan", "1")))
            except Exception:
                self._colspan = 1

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            # Join with a space, not "": handle_data() fires once per
            # contiguous text run, and multiple runs only happen because a
            # tag (e.g. a <br> the table engine inserted for a wrapped/
            # multi-line cell) sat between them - joining with "" glues two
            # separate lines together with no boundary at all. The \s+
            # collapse below absorbs any resulting double space.
            cell = re.sub(r"\s+", " ", " ".join(self._cell)).strip()
            self._row.append(cell)
            for _ in range(self._colspan - 1):
                self._row.append("")
            self._cell = []
            self._in_cell = False
            self._colspan = 1
        elif tag == "tr":
            self.rows.append(self._row)
            self._row = []

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


def _html_to_grid(html: str) -> List[List[str]]:
    if not html:
        return []
    p = _TableGridExtractor()
    try:
        p.feed(html)
    except Exception:
        return []
    return p.rows


def _table_to_grid(table: Dict[str, Any]) -> List[List[str]]:
    grid = _html_to_grid(table.get("html") or "")
    if grid:
        return grid
    headers = table.get("headers") or []
    rows = table.get("rows") or []
    if headers or rows:
        out: List[List[str]] = []
        if headers:
            out.append([str(c or "") for c in headers])
        for r in rows:
            out.append([str(c or "") for c in (r or [])])
        return out
    return []


def _ink_ratio(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    return float(np.sum(gray < 140)) / float(gray.size)


def _crop_gray(img_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), dtype=np.uint8)
    crop = img_rgb[y1:y2, x1:x2]
    if crop.ndim == 3:
        return (
            0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]
        ).astype(np.uint8)
    return crop


def _window_ink_score(img_rgb: np.ndarray, cx1: int, cy1: int, cx2: int, cy2: int) -> float:
    gray = _crop_gray(img_rgb, cx1, cy1, cx2, cy2)
    if gray.size < 16:
        return 0.0
    gh, gw = gray.shape[:2]
    if gh >= 6 and gw >= 6:
        inner = gray[gh // 5 : 4 * gh // 5, gw // 5 : 4 * gw // 5]
        # Interior weight: checked X/fill is dark in the middle; empty boxes are edge-only
        return 0.30 * _ink_ratio(gray) + 0.70 * _ink_ratio(inner)
    return _ink_ratio(gray)


def _scan_checkbox_left(
    img_rgb: Optional[np.ndarray],
    text_bbox: List[int],
    lo: int = 10,
    hi: int = 42,
) -> Optional[float]:
    """
    Scan horizontal offsets left of a label for the checkbox square.
    Forms often leave 15–35px between the box and the word.
    """
    if img_rgb is None or not text_bbox or len(text_bbox) < 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in text_bbox[:4]]
    th = max(8, y2 - y1)
    box = max(10, min(18, th))
    cy = (y1 + y2) // 2
    cy1 = cy - box // 2
    cy2 = cy1 + box
    best = 0.0
    for dx in range(lo, hi + 1, 2):
        cx2 = x1 - dx
        cx1 = cx2 - box
        best = max(best, _window_ink_score(img_rgb, cx1, cy1, cx2, cy2))
    return best


def _scan_checkbox_between(
    img_rgb: Optional[np.ndarray],
    yes_bb: List[int],
    no_bb: List[int],
) -> Optional[float]:
    """Score the checkbox that sits between adjacent Yes and No labels."""
    if img_rgb is None or len(yes_bb) < 4 or len(no_bb) < 4:
        return None
    y1 = min(int(yes_bb[1]), int(no_bb[1]))
    y2 = max(int(yes_bb[3]), int(no_bb[3]))
    box = max(10, min(18, y2 - y1))
    cy = (y1 + y2) // 2
    cy1 = cy - box // 2
    cy2 = cy1 + box
    # Stay to the right of the Yes glyph center so we don't re-hit the Yes checkbox
    left_bound = (int(yes_bb[0]) + int(yes_bb[2])) // 2
    right_bound = int(no_bb[0]) - 1
    if right_bound - left_bound < box + 2:
        # Tiny gap: probe a few px left of No only
        return _scan_checkbox_left(img_rgb, no_bb, lo=2, hi=12)
    best = 0.0
    for cx2 in range(left_bound + box, right_bound + 1, 2):
        cx1 = cx2 - box
        best = max(best, _window_ink_score(img_rgb, cx1, cy1, cx2, cy2))
    return best


def _checkbox_ink_score(img_rgb: Optional[np.ndarray], text_bbox: List[int]) -> Optional[float]:
    """Best ink score for a checkbox left of a label (scanned offsets)."""
    return _scan_checkbox_left(img_rgb, text_bbox)


def _checkbox_checked(img_rgb: Optional[np.ndarray], text_bbox: List[int]) -> Optional[bool]:
    """Estimate if a checkbox to the LEFT of a Yes/No label is filled."""
    score = _checkbox_ink_score(img_rgb, text_bbox)
    if score is None:
        return None
    if score >= 0.22:
        return True
    if score <= 0.12:
        return False
    return score >= 0.17


def _standalone_checkbox_checked(
    img_rgb: Optional[np.ndarray], text_bbox: List[int]
) -> bool:
    """Best-effort checked/unchecked probe for a standalone checkbox glyph
    immediately preceding a bare "<...> Indicator" label (e.g. "EOB
    Indicator").

    Screenshot-rendered UI checkboxes sit only a few px from their label -
    closer, and with a much fainter border, than the scanned-paper spacing/
    darkness threshold _scan_checkbox_left assumes - and PaddleOCR often
    folds the checkbox glyph into the same detection box as the label text
    instead of giving it a separate box, so the probe window is anchored
    from the label's own bbox (scaled to its text height) rather than
    scanned to its left. Defaults to unchecked (False) whenever no clear
    box border is found at that offset: the caller already knows (from the
    text match) that a checkbox exists here, and unset is by far the most
    common state for a UI checkbox, so "can't confirm a fill" should read
    as explicit False rather than dropping the field.
    """
    if img_rgb is None or not text_bbox or len(text_bbox) < 4:
        return False
    x1, y1, x2, y2 = [int(v) for v in text_bbox[:4]]
    th = max(8, y2 - y1)
    box = int(max(6, min(14, th * 0.5)))
    dx = int(max(6, min(16, th * 0.75)))
    cy = (y1 + y2) // 2
    cy1 = cy - box // 2
    cy2 = cy1 + box
    cx2 = x1 + dx
    cx1 = cx2 - box
    if cx1 < 0:
        return False
    gray = _crop_gray(img_rgb, cx1, cy1, cx2, cy2)
    if gray.size < 9:
        return False
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    perim_ink = float(np.mean(gray[mask] < 180))
    if perim_ink < 0.30:
        return False
    inner = gray[1:-1, 1:-1]
    inner_ink = float(np.mean(inner < 180)) if inner.size else 0.0
    return inner_ink >= 0.35


def _detect_standalone_indicator_checkboxes(
    words: List[Dict[str, Any]], image_rgb: Optional[np.ndarray]
) -> List[Dict[str, Any]]:
    """Standalone "<...> Indicator" checkbox fields (e.g. EOB/MPPR Indicator).

    Unlike detect_checkboxes()'s Yes/No handling, these have a single box
    and no adjacent Yes/No text to anchor against - text pattern gates
    which words this even looks at (see _STANDALONE_INDICATOR_RE), so the
    checked-state probe never runs against unrelated labels/headings.
    """
    out: List[Dict[str, Any]] = []
    for w in words:
        raw = (w.get("text") or "").strip()
        if not raw or ":" in raw:
            continue
        if not _STANDALONE_INDICATOR_RE.match(raw):
            continue
        bbox = list(w.get("bbox") or [])
        checked = _standalone_checkbox_checked(image_rgb, bbox)
        out.append(
            {
                "label": raw,
                "value": "Yes" if checked else "No",
                "checked": checked,
                "bbox": bbox,
                "source": "standalone_indicator",
            }
        )
    return out


def _resolve_yes_no(
    img_rgb: Optional[np.ndarray],
    yes_bb: Optional[List[int]],
    no_bb: Optional[List[int]],
) -> Tuple[str, Optional[bool], Optional[bool], Optional[float], Optional[float]]:
    """
    Pick Yes/No by comparing scanned ink scores.
    Yes box is farther left of 'Yes'; No box sits between the two labels.
    """
    yes_s = _scan_checkbox_left(img_rgb, yes_bb or [], lo=12, hi=42)
    if yes_bb and no_bb and (no_bb[0] - yes_bb[0]) < 120:
        no_s = _scan_checkbox_between(img_rgb, yes_bb, no_bb)
    else:
        no_s = _scan_checkbox_left(img_rgb, no_bb or [], lo=8, hi=28)

    yes_state = None if yes_s is None else yes_s >= 0.18
    no_state = None if no_s is None else no_s >= 0.18

    if yes_s is not None and no_s is not None:
        if yes_s >= no_s + 0.04:
            return "Yes", True, False, yes_s, no_s
        if no_s >= yes_s + 0.04:
            return "No", False, True, yes_s, no_s
        if yes_s >= 0.20 and no_s < 0.16:
            return "Yes", True, False, yes_s, no_s
        if no_s >= 0.20 and yes_s < 0.16:
            return "No", False, True, yes_s, no_s
        return "Unknown", yes_state, no_state, yes_s, no_s

    if yes_state is True and no_state is not True:
        return "Yes", yes_state, no_state, yes_s, no_s
    if no_state is True and yes_state is not True:
        return "No", yes_state, no_state, yes_s, no_s
    return "Unknown", yes_state, no_state, yes_s, no_s


def _iter_word_dicts(page: PageResult) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ln in page.lines or []:
        if ln.words:
            for w in ln.words:
                out.append({"text": w.text, "bbox": list(w.bbox), "conf": w.conf})
        elif ln.text:
            out.append({"text": ln.text, "bbox": list(ln.bbox), "conf": ln.conf})
    return out


def _valid_option_bbox(bb: List[int], page: PageResult) -> bool:
    """Reject table-sized OCR boxes that are not real Yes/No labels."""
    if not bb or len(bb) < 4:
        return False
    w = bb[2] - bb[0]
    h = bb[3] - bb[1]
    if h <= 0 or w <= 0:
        return False
    if h > 48 or w > 160:
        return False
    pw = page.width or 1
    ph = page.height or 1
    if w * h > pw * ph * 0.015:
        return False
    return True


def _split_yes_no_token(w: Dict[str, Any], page: PageResult) -> List[Dict[str, Any]]:
    t = (w.get("text") or "").strip()
    bb = w.get("bbox") or [0, 0, 0, 0]
    if not _valid_option_bbox(bb, page):
        return []
    if re.fullmatch(r"Yes", t, re.I):
        return [{"text": "Yes", "bbox": bb, "conf": w.get("conf", 0)}]
    if re.fullmatch(r"No", t, re.I):
        return [{"text": "No", "bbox": bb, "conf": w.get("conf", 0)}]
    if re.fullmatch(r"Yes\s*[/|]?\s*No", t, re.I) or t.lower() in {"yes no", "yes/no"}:
        mid = (bb[0] + bb[2]) // 2
        return [
            {"text": "Yes", "bbox": [bb[0], bb[1], mid, bb[3]], "conf": w.get("conf", 0)},
            {"text": "No", "bbox": [mid, bb[1], bb[2], bb[3]], "conf": w.get("conf", 0)},
        ]
    # Jammed cell: "... Translation Services Available. Yes No"
    if re.search(r"\bYes\s+No\b", t, re.I) and re.search(r"Translation", t, re.I):
        # Place synthetic Yes/No just inside the right edge of the cell
        y1, y2 = bb[1], bb[3]
        # Prefer lower band of cell (form row is usually at bottom of tall cells)
        if y2 - y1 > 40:
            y1 = y2 - 22
        x2 = bb[2]
        return [
            {"text": "Yes", "bbox": [x2 - 70, y1, x2 - 38, y2], "conf": w.get("conf", 0)},
            {"text": "No", "bbox": [x2 - 36, y1, x2 - 4, y2], "conf": w.get("conf", 0)},
        ]
    return []


def _find_translation_anchors(
    words: List[Dict[str, Any]],
    page: PageResult,
    yes_no_tokens: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Find Translation Services Available anchors from words or jammed line/table text."""
    anchors: List[Dict[str, Any]] = []
    mid = (page.width or 0) / 2.0

    # Multi-word sequence (skip oversized boxes)
    texts = [(w.get("text") or "").strip() for w in words]
    for i in range(len(words)):
        window = " ".join(texts[i : i + 4])
        if not re.search(r"Translation\s+Services\s+Available", window, re.I):
            continue
        bbs = [
            words[j]["bbox"]
            for j in range(i, min(i + 4, len(words)))
            if _valid_option_bbox(words[j].get("bbox") or [], page)
            or (
                words[j].get("bbox")
                and (words[j]["bbox"][3] - words[j]["bbox"][1]) < 40
                and (words[j]["bbox"][2] - words[j]["bbox"][0]) < page.width * 0.55
            )
        ]
        if bbs:
            bbox = [
                min(b[0] for b in bbs),
                min(b[1] for b in bbs),
                max(b[2] for b in bbs),
                max(b[3] for b in bbs),
            ]
            if bbox[3] - bbox[1] < 80:
                anchors.append({"bbox": bbox, "text": "Translation Services Available"})

    # Single jammed token / line (must be compact)
    for w in words:
        t = w.get("text") or ""
        bb = w.get("bbox") or []
        if not re.search(r"Translation\s+Services\s+Available", t, re.I):
            continue
        if len(bb) >= 4 and (bb[3] - bb[1]) < 60 and (bb[2] - bb[0]) < page.width * 0.7:
            anchors.append({"bbox": bb, "text": "Translation Services Available"})

    # Fallback: one left + one right from Yes/No tokens in lower page band.
    # Only fires when the page actually has Yes/No text somewhere - without
    # that check this used to fire on *every* form with no "Translation
    # Services Available" field at all, fabricating two "Unknown" checkbox
    # lines (left/right) into pages that never had this field to begin with.
    # Yes/No text alone isn't enough either: an ordinary value like "Board
    # Cert: YES" also matches, so the page must actually mention the field
    # and offer both options.
    page_mentions_field = bool(
        re.search(r"translation", page.text or "", re.I)
        or any(re.search(r"translation", (w.get("text") or ""), re.I) for w in words)
    )
    has_both_options = bool(yes_no_tokens) and {
        (t.get("text") or "").lower() for t in yes_no_tokens
    } >= {"yes", "no"}
    if not anchors and page_mentions_field and has_both_options:
        lower_y = (page.height or 0) * 0.55
        for side, x_min, x_max in (
            ("left", 0, mid),
            ("right", mid, page.width or mid * 2),
        ):
            anchors.append(
                {
                    "bbox": [int(x_min + 10), int(lower_y), int(x_max - 10), int(page.height or lower_y)],
                    "text": "Translation Services Available",
                    "side": side,
                    "from_table": True,
                }
            )

    # Dedup by approximate y/side
    unique: List[Dict[str, Any]] = []
    for a in anchors:
        bb = a.get("bbox") or [0, 0, 0, 0]
        y = (bb[1] + bb[3]) / 2.0 if len(bb) >= 4 else 0
        side = a.get("side") or ("left" if bb[0] < mid else "right")
        key = (round(y / 40), side)
        if any(
            (
                round(
                    (
                        (u.get("bbox") or [0, 0, 0, 0])[1]
                        + (u.get("bbox") or [0, 0, 0, 0])[3]
                    )
                    / 2.0
                    / 40
                ),
                u.get("side")
                or ("left" if (u.get("bbox") or [0])[0] < mid else "right"),
            )
            == key
            for u in unique
        ):
            continue
        a["side"] = side
        unique.append(a)
    return unique


def detect_checkboxes(
    page: PageResult,
    image_rgb: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Detect Yes/No checkbox states for form fields."""
    words = _iter_word_dicts(page)
    results: List[Dict[str, Any]] = []

    # Unicode marks
    for w in words:
        t = w.get("text") or ""
        for ch in t:
            if ch in _CHECK_MARKS_CHECKED or ch in _CHECK_MARKS_UNCHECKED:
                label = t.replace(ch, "").strip(" :.-")
                results.append(
                    {
                        "label": label or "Checkbox",
                        "value": "Yes" if ch in _CHECK_MARKS_CHECKED else "No",
                        "checked": ch in _CHECK_MARKS_CHECKED,
                        "bbox": w.get("bbox") or [],
                        "source": "unicode",
                    }
                )

    yes_no_tokens: List[Dict[str, Any]] = []
    for w in words:
        yes_no_tokens.extend(_split_yes_no_token(w, page))

    anchors = _find_translation_anchors(words, page, yes_no_tokens)
    mid = (page.width or 0) / 2.0

    for lab in anchors:
        lb = lab.get("bbox") or [0, 0, 0, 0]
        side = lab.get("side") or ("left" if lb[0] < mid else "right")
        ly = (lb[1] + lb[3]) / 2.0 if len(lb) >= 4 and lb[3] > lb[1] else None

        candidates = []
        for t in yes_no_tokens:
            bb = t.get("bbox") or [0, 0, 0, 0]
            if not _valid_option_bbox(bb, page):
                continue
            tx = (bb[0] + bb[2]) / 2.0
            token_side = "left" if tx < mid else "right"
            if token_side != side:
                continue
            if ly is not None and not lab.get("from_table"):
                ty = (bb[1] + bb[3]) / 2.0
                if abs(ty - ly) > 55:
                    continue
            if lab.get("from_table"):
                # Lower half of page only for fallback anchors
                if bb[1] < (page.height or 0) * 0.5:
                    continue
            candidates.append(t)

        if ly is not None and not lab.get("from_table"):
            candidates.sort(
                key=lambda t: abs(
                    (
                        (t.get("bbox") or [0, 0, 0, 0])[1]
                        + (t.get("bbox") or [0, 0, 0, 0])[3]
                    )
                    / 2.0
                    - ly
                )
            )
        else:
            candidates.sort(key=lambda t: (t.get("bbox") or [0, 0])[1])

        yes_bb = None
        no_bb = None
        seen_yes = False
        seen_no = False
        for t in candidates:
            if re.fullmatch(r"Yes", t.get("text") or "", re.I) and not seen_yes:
                yes_bb = t.get("bbox")
                seen_yes = True
            elif re.fullmatch(r"No", t.get("text") or "", re.I) and not seen_no:
                no_bb = t.get("bbox")
                seen_no = True
            if seen_yes and seen_no:
                break

        value, yes_state, no_state, yes_s, no_s = _resolve_yes_no(
            image_rgb, yes_bb, no_bb
        )

        results.append(
            {
                "label": "Translation Services Available",
                "value": value,
                "checked_yes": yes_state,
                "checked_no": no_state,
                "yes_ink_score": yes_s,
                "no_ink_score": no_s,
                "bbox": lb,
                "yes_bbox": yes_bb or [],
                "no_bbox": no_bb or [],
                "source": "image_ink" if image_rgb is not None else "text",
                "column": side,
            }
        )

    results.extend(_detect_standalone_indicator_checkboxes(words, image_rgb))

    # Deduplicate by label+column
    seen = set()
    unique: List[Dict[str, Any]] = []
    for r in results:
        key = (r.get("label"), r.get("column"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def _is_label_text(text: str) -> bool:
    t = (text or "").strip().rstrip(":").lower()
    if not t:
        return False
    return any(t == l.lower() or t.startswith(l.lower()) for l in _FORM_LABELS)


_GENERIC_LABEL_VALUE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /#().'_%-]{1,60}?)\s*:\s*(.+)$")


def _split_label_value(cell: str) -> Tuple[str, str]:
    """Text-only label/value split (no word geometry available here).

    Tries the generic "Label: value" shape first - any colon-bearing cell
    splits correctly regardless of what the label says, so this works on
    forms this parser has never seen. Only falls back to the known-label
    vocabulary for cells with no colon at all (jammed OCR like "Suite#101"
    or "NPI 1234567890"), where a boundary can't be found generically.
    """
    c = re.sub(r"\s+", " ", (cell or "")).strip()
    if not c:
        return "", ""
    m = _GENERIC_LABEL_VALUE_RE.match(c)
    if m:
        label, value = m.group(1).strip(), m.group(2).strip()
        # "Street Address Suite #" → both labels, not label/value
        if value and _is_label_text(value):
            return f"{label} / {value}", ""
        return label, value
    # Fallback layer: no colon present - recognize the label from known
    # form vocabulary instead of splitting on geometry/punctuation.
    m2 = _LABEL_SPLIT_RE.match(c)
    if m2:
        label = m2.group("label").strip()
        value = m2.group("value").strip(" .:")
        if value and _is_label_text(value):
            return f"{label} / {value}", ""
        return label, value
    return c, ""


def _split_label_value_geom(words: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Primary label/value splitter: uses word geometry, not a fixed label list.

    Two form-agnostic signals, tried in order:
      1. A colon token (or a word ending in ':') - the clearest possible
         boundary, independent of what the label text actually says.
      2. The single horizontal gap between adjacent words that clearly
         stands out from the row's other gaps - forms routinely leave a
         wide run of blank space (or dot leaders) between a label and its
         answer, no matter the label's wording. The gap floor is derived
         from this row's own median word height, not a fixed pixel value,
         so it adapts to the page's resolution/font size.

    Falls back to the known-label text matcher only when the cell is a
    single OCR token (nothing to split geometrically) or neither signal
    above finds an unambiguous boundary - e.g. a jammed token or a label
    typed with an ordinary single-space gap.
    """
    ws = [w for w in words if (w.get("text") or "").strip() and w.get("bbox")]
    if len(ws) < 2:
        joined = " ".join((w.get("text") or "").strip() for w in ws)
        return _split_label_value(joined)

    ordered = sorted(ws, key=lambda w: w["bbox"][0])

    for i, w in enumerate(ordered):
        t = (w.get("text") or "").strip()
        if t == ":" or (t.endswith(":") and len(t) > 1):
            label = " ".join((x.get("text") or "").strip() for x in ordered[: i + 1]).strip().rstrip(":").strip()
            value = " ".join((x.get("text") or "").strip() for x in ordered[i + 1 :]).strip()
            if label:
                return label, value

    gaps = [(ordered[i]["bbox"][0] - ordered[i - 1]["bbox"][2], i) for i in range(1, len(ordered))]
    heights = sorted(max(1, w["bbox"][3] - w["bbox"][1]) for w in ordered)
    med_h = heights[len(heights) // 2]
    gap_floor = max(6.0, med_h * 0.8)

    gap_vals = [g for g, _ in gaps]
    ranked = sorted((g for g in gaps if g[0] >= gap_floor), key=lambda g: -g[0])
    if ranked:
        max_gap, split_at = ranked[0]
        other = [g for g in gap_vals if g != max_gap]
        is_outlier = not other or max_gap >= 2.0 * statistics.median(other)
        if is_outlier:
            label = " ".join((x.get("text") or "").strip() for x in ordered[:split_at]).strip()
            value = " ".join((x.get("text") or "").strip() for x in ordered[split_at:]).strip()
            if label and any(ch.isalpha() for ch in label):
                return label, value

    joined = " ".join((w.get("text") or "").strip() for w in ordered)
    return _split_label_value(joined)


def _words_to_rows(
    words: List[Dict[str, Any]], y_tol: Optional[float] = None
) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    if y_tol is None:
        # Adapt to this page's own text size instead of a fixed pixel value
        # tuned for one resolution - a 14px tolerance is loose for a small
        # scan and tight for a high-res screenshot with large fonts.
        heights = sorted(
            max(1, (w["bbox"][3] - w["bbox"][1])) for w in words if w.get("bbox")
        )
        med_h = heights[len(heights) // 2] if heights else 12.0
        y_tol = max(6.0, med_h * 0.6)
    ordered = sorted(
        words,
        key=lambda w: (
            ((w.get("bbox") or [0, 0, 0, 0])[1] + (w.get("bbox") or [0, 0, 0, 0])[3]) // 2,
            (w.get("bbox") or [0])[0],
        ),
    )
    rows: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_y = None
    for w in ordered:
        bb = w.get("bbox") or [0, 0, 0, 0]
        y = (bb[1] + bb[3]) // 2
        if cur_y is None or abs(y - cur_y) <= y_tol:
            cur.append(w)
            cur_y = y if cur_y is None else (cur_y + y) // 2
        else:
            rows.append(cur)
            cur = [w]
            cur_y = y
    if cur:
        rows.append(cur)
    return rows


def _existing_multi_col_table_bands(page: PageResult, min_cols: int = 3) -> List[Tuple[float, float]]:
    """Y-ranges already covered by a genuine N-column data table (headers >= min_cols).

    rebuild_two_column_form_table() below force-splits every row on the page at
    the horizontal midpoint. That's correct for an actual 2-column label/value
    form, but a wide multi-column grid (e.g. a 5-column CPT/Modifier/Description/
    Diag Pointer/Date table) has 3+ populated fields per row - the midpoint split
    lumps whichever of them land left-of-midpoint into one cell, and whether a
    given row's real columns survive intact ends up depending on which fields
    happen to be empty (e.g. a blank Modifier) rather than on genuine column
    boundaries. Excluding rows already covered by such a table keeps the 2-column
    rebuild scoped to the parts of the page that are actually 2-column forms.
    """
    bands: List[Tuple[float, float]] = []
    for t in page.tables or []:
        if not isinstance(t, dict):
            continue
        headers = t.get("headers") or []
        bbox = t.get("bbox") or []
        if len(headers) >= min_cols and len(bbox) >= 4:
            bands.append((bbox[1], bbox[3]))
    return bands


def _in_bands(y: float, bands: List[Tuple[float, float]]) -> bool:
    return any(top <= y <= bot for top, bot in bands)


def _row_clusters(row_words: List[Dict[str, Any]], gap_floor: float) -> List[Dict[str, Any]]:
    """Group one row's words into clusters separated by a gap >= gap_floor.

    Same idea as the label/value gap split in _split_label_value_geom, but run
    across the whole row instead of stopping at the first split - that's what
    lets a wide N-column data row (unlike a simple "label: value" line) reveal
    all of its column boundaries at once.
    """
    ordered = sorted(
        (w for w in row_words if (w.get("text") or "").strip() and w.get("bbox")),
        key=lambda w: w["bbox"][0],
    )
    if not ordered:
        return []
    clusters: List[List[Dict[str, Any]]] = [[ordered[0]]]
    for w in ordered[1:]:
        prev = clusters[-1][-1]
        if w["bbox"][0] - prev["bbox"][2] >= gap_floor:
            clusters.append([w])
        else:
            clusters[-1].append(w)
    out = []
    for c in clusters:
        x0 = min(x["bbox"][0] for x in c)
        x1 = max(x["bbox"][2] for x in c)
        out.append({
            "words": c,
            "x0": x0,
            "x1": x1,
            "xc": (x0 + x1) / 2.0,
            "text": " ".join((x.get("text") or "").strip() for x in c).strip(),
        })
    return out


def _detect_wide_grids(
    rows: List[List[Dict[str, Any]]], page_width: float, gap_floor: float
) -> List[Tuple[int, int, List[Dict[str, Any]]]]:
    """Find contiguous runs of rows that form a genuine N-column (N>=3) data
    grid - e.g. a Seq/CPT-HCPCS/Modifier/Description/Diag Pointer/Date
    procedure table - that reconstruct_tables()/PPStructure didn't tag as a
    table upstream. rebuild_two_column_form_table() force-splits every row at
    the page's horizontal midpoint, which only makes sense for an actual
    2-column label/value form; finding these runs first lets it skip them
    instead of merging 3+ real fields into one cell per side (which is what
    silently flattened the procedure table - whether a row's columns survived
    depended on which fields happened to be blank, e.g. an empty Modifier).

    Returns a list of (start_row_idx, end_row_idx_exclusive, header_clusters).
    """
    clustered = [_row_clusters(r, gap_floor) for r in rows]
    blocks: List[Tuple[int, int, List[Dict[str, Any]]]] = []
    i = 0
    n = len(rows)
    while i < n:
        header = clustered[i]
        # A real column-header row (e.g. "Seq | CPT/HCPCS | Modifier | ...")
        # is a run of short titles with no colon. A 2-column form's field row
        # (e.g. "Diagnosis Code Type: ICD-10-CM   POA Indicator: Y") also
        # clusters into 3+ wide-spread groups, but its clusters carry the
        # label's colon - that's what tells the two apart, not phrasing.
        has_colon = any(":" in c["text"] for c in header)
        if (
            len(header) < 3
            or (header[-1]["x1"] - header[0]["x0"]) < page_width * 0.5
            or has_colon
        ):
            i += 1
            continue
        # A continuation row must actually line up with the header's column
        # anchors, not merely be wide - a page's top nav bar/title strip
        # (e.g. "Curamind Claims Management | Sarah Johnson | ... | Logout"
        # followed by a page title, then a tab strip) also spreads 3+ short
        # clusters across the width but the rows below it don't share its
        # column positions, unlike a real grid's data rows.
        anchors = [c["xc"] for c in header]
        align_tol = max(page_width * 0.08, gap_floor * 3)
        j = i + 1
        while j < n:
            row_cl = clustered[j]
            if len(row_cl) < 2:
                break
            aligned = sum(
                1 for c in row_cl
                if min(abs(c["xc"] - a) for a in anchors) <= align_tol
            )
            if aligned / len(row_cl) < 0.6:
                break
            j += 1
        if j - i >= 2:
            blocks.append((i, j, header))
            i = j
        else:
            i += 1
    return blocks


def _build_wide_grid_table(
    rows: List[List[Dict[str, Any]]], start: int, end: int, header_clusters: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build a table dict for one _detect_wide_grids() block.

    Assigns each data-row word to its nearest header column anchor (mirrors
    word_to_col() in reconstruct_tables(), ocr/layout.py), so a row with a
    blank middle column (e.g. no Modifier) still lands its other fields in
    the right cells instead of shifting them left.
    """
    anchors = [c["xc"] for c in header_clusters]
    n_cols = len(anchors)

    def word_to_col(bbox: List[float]) -> int:
        xc = (bbox[0] + bbox[2]) / 2.0
        return min(range(n_cols), key=lambda k: abs(anchors[k] - xc))

    headers = [c["text"] for c in header_clusters]
    data_rows: List[List[str]] = []
    all_words: List[Dict[str, Any]] = list(header_clusters and [w for c in header_clusters for w in c["words"]])
    for ri in range(start + 1, end):
        cells: List[List[str]] = [[] for _ in range(n_cols)]
        row_words = sorted(
            (w for w in rows[ri] if (w.get("text") or "").strip() and w.get("bbox")),
            key=lambda w: w["bbox"][0],
        )
        for w in row_words:
            cells[word_to_col(w["bbox"])].append(w["text"].strip())
            all_words.append(w)
        data_rows.append([" ".join(c).strip() for c in cells])

    html_parts = ["<table>"]
    html_parts.append("<tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr>")
    for r in data_rows:
        html_parts.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>")
    html_parts.append("</table>")

    xs = [w["bbox"][0] for w in all_words]
    ys = [w["bbox"][1] for w in all_words]
    xe = [w["bbox"][2] for w in all_words]
    ye = [w["bbox"][3] for w in all_words]

    return {
        "bbox": [min(xs), min(ys), max(xe), max(ye)],
        "html": "".join(html_parts),
        "text": "\n".join(" | ".join(r) for r in data_rows),
        "headers": headers,
        "rows": data_rows,
        "source": "wide_grid",
        "engine": "form_postprocess",
        "confidence": 0.85,
    }


def rebuild_two_column_form_table(page: PageResult) -> Optional[List[Dict[str, Any]]]:
    """Rebuild cleaner table(s) from OCR word geometry.

    Returns any wide (3+ column) data grids found via _detect_wide_grids(),
    plus a 2-column form table built from whatever rows are left over - or
    None if neither could be built.
    """
    exclude_bands = _existing_multi_col_table_bands(page)
    words = [
        w
        for w in _iter_word_dicts(page)
        if (w.get("text") or "").strip()
        and len(w.get("bbox") or []) >= 4
        and (w["bbox"][2] - w["bbox"][0]) < (page.width or 1) * 0.85
        and not _in_bands((w["bbox"][1] + w["bbox"][3]) / 2.0, exclude_bands)
    ]
    if len(words) < 8 or not page.width:
        return None

    all_rows = _words_to_rows(words)

    heights = sorted(max(1, w["bbox"][3] - w["bbox"][1]) for w in words)
    med_h = heights[len(heights) // 2]
    grid_gap_floor = max(6.0, med_h * 0.8)

    grid_blocks = _detect_wide_grids(all_rows, page.width, grid_gap_floor)
    grid_row_idx = {idx for start, end, _ in grid_blocks for idx in range(start, end)}
    wide_tables = [
        _build_wide_grid_table(all_rows, start, end, header)
        for start, end, header in grid_blocks
    ]

    mid = page.width / 2.0
    rows = [r for idx, r in enumerate(all_rows) if idx not in grid_row_idx]
    word_rows: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    grid: List[List[str]] = []
    for row in rows:
        left = sorted([w for w in row if w["bbox"][0] < mid], key=lambda w: w["bbox"][0])
        right = sorted([w for w in row if w["bbox"][0] >= mid], key=lambda w: w["bbox"][0])
        if not left and not right:
            continue
        word_rows.append((left, right))
        grid.append(
            [
                " ".join(w["text"].strip() for w in left).strip(),
                " ".join(w["text"].strip() for w in right).strip(),
            ]
        )

    if len(grid) < 4:
        return wide_tables or None

    # Keep geometric left|right alignment; split label/value inside each cell
    # from word geometry (colon token + horizontal gap - see
    # _split_label_value_geom). Carry forward pending labels when the next
    # row is value-only; the known-label list only breaks the tie when
    # geometry alone can't say whether an unsplit row is itself a fresh
    # label or a value that continues the previous one (_is_label_text).
    paired_grid: List[List[str]] = [
        ["Left Field", "Left Value", "Right Field", "Right Value"]
    ]
    pending_left = ""
    pending_right = ""
    for left_words, right_words in word_rows:
        left_t = " ".join(w["text"].strip() for w in left_words).strip()
        right_t = " ".join(w["text"].strip() for w in right_words).strip()
        lf, lv = _split_label_value_geom(left_words)
        rf, rv = _split_label_value_geom(right_words)

        # Value-only row fills previous empty label
        if pending_left and not lv and lf == left_t and not _is_label_text(left_t):
            lf, lv = pending_left, left_t
            pending_left = ""
        elif lv:
            pending_left = ""
        elif lf and not lv and _is_label_text(left_t):
            pending_left = lf
        else:
            pending_left = ""

        if pending_right and not rv and rf == right_t and not _is_label_text(right_t):
            rf, rv = pending_right, right_t
            pending_right = ""
        elif rv:
            pending_right = ""
        elif rf and not rv and _is_label_text(right_t):
            pending_right = rf
        else:
            pending_right = ""

        # Skip pure label rows that will be filled by next value row
        if (pending_left == lf and not lv) or (pending_right == rf and not rv):
            # still emit if the other side has a complete pair
            if not ((lv or (lf and lf != pending_left)) or (rv or (rf and rf != pending_right))):
                continue

        if not lf and not lv and not rf and not rv:
            continue
        paired_grid.append([lf, lv, rf, rv])

    html_parts = ["<table>"]
    for i, row in enumerate(paired_grid):
        tag = "th" if i == 0 else "td"
        html_parts.append("<tr>" + "".join(f"<{tag}>{escape(c)}</{tag}>" for c in row) + "</tr>")
    html_parts.append("</table>")

    form_words = [w for left, right in word_rows for w in (left + right)]
    ys = [w["bbox"][1] for w in form_words]
    xs = [w["bbox"][0] for w in form_words]
    ye = [w["bbox"][3] for w in form_words]
    xe = [w["bbox"][2] for w in form_words]

    kv_lines: List[str] = []
    for lf, lv, rf, rv in paired_grid[1:]:
        if lf or lv:
            kv_lines.append(f"{lf}: {lv}" if lf else lv)
        if rf or rv:
            kv_lines.append(f"{rf}: {rv}" if rf else rv)

    form_table = {
        "bbox": [min(xs), min(ys), max(xe), max(ye)],
        "html": "".join(html_parts),
        "text": "\n".join(kv_lines),
        "headers": paired_grid[0],
        "rows": paired_grid[1:],
        "raw_two_col": grid,
        "source": "two_column_form",
        "engine": "form_postprocess",
        "confidence": 0.92,
    }
    return [form_table] + wide_tables


def clean_ppstructure_table(table: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize PPStructure cells: compress to 2 cols, split known labels.

    Skips the 2-column compression for a table already reconstructed with proper
    N-column structure (source == "layout", from reconstruct_tables() in
    ocr/layout.py) - those already have one field per column. Running such a table
    through the left-half/right-half compression below would jam unrelated fields
    together, and the label/value expansion after it keeps only 2 of them - this is
    what flattened a 6-column CPT/Modifier/Description/Diag Pointer/Date procedure
    table into a single blob per row.
    """
    t = dict(table)

    if (table.get("source") or "") == "layout":
        headers = [re.sub(r"\s+", " ", (c or "")).strip() for c in (table.get("headers") or [])]
        rows = [
            [re.sub(r"\s+", " ", (c or "")).strip().replace("..", ".") for c in row]
            for row in (table.get("rows") or [])
        ]
        html_parts = ["<table>"]
        if headers:
            html_parts.append("<tr>" + "".join(f"<th>{escape(c)}</th>" for c in headers) + "</tr>")
        for row in rows:
            html_parts.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>")
        html_parts.append("</table>")
        t["headers"] = headers
        t["rows"] = rows
        t["html"] = "".join(html_parts)
        t["text"] = "\n".join(" | ".join(r) for r in rows)
        t["cleaned"] = True
        return t

    grid = _table_to_grid(t)
    if not grid:
        return t

    cleaned: List[List[str]] = []
    for row in grid:
        cells = [re.sub(r"\s+", " ", (c or "")).strip().replace("..", ".") for c in row]
        if any(cells):
            cleaned.append(cells)

    if cleaned:
        cols = max(len(r) for r in cleaned)
        if cols >= 4:
            compressed = []
            for row in cleaned:
                row = row + [""] * (cols - len(row))
                left_parts = [c for c in row[: cols // 2] if c]
                right_parts = [c for c in row[cols // 2 :] if c]
                compressed.append(
                    [" ".join(left_parts).strip(), " ".join(right_parts).strip()]
                )
            cleaned = compressed

    # Expand jammed cells into Label | Value within each side
    expanded: List[List[str]] = [["Left Field", "Left Value", "Right Field", "Right Value"]]
    for row in cleaned:
        left = row[0] if row else ""
        right = row[1] if len(row) > 1 else ""
        # Pull Translation / Yes No into structured fields
        for side_name, cell in (("left", left), ("right", right)):
            if re.search(r"Translation\s+Services\s+Available", cell, re.I):
                # strip email before translation if present
                cell = re.sub(
                    r"(Translation\s+Services\s+Available)\.?\s*(Yes\s*No)?",
                    r"\1",
                    cell,
                    flags=re.I,
                )
                if side_name == "left":
                    left = cell
                else:
                    right = cell

        lf, lv = _split_label_value(left)
        rf, rv = _split_label_value(right)
        # If whole cell is "email Translation..." split email out
        email_m = re.search(r"([\w.+-]+@[\w.-]+\.\w+)", left)
        if email_m and "translation" in left.lower():
            lv = email_m.group(1)
            lf = "Email"
            # translation handled via checkboxes
        email_m2 = re.search(r"([\w.+-]+@[\w.-]+\.\w+)", right)
        if email_m2 and "translation" in right.lower():
            rv = email_m2.group(1)
            rf = "Email"
        expanded.append([lf, lv, rf, rv])

    html_parts = ["<table>"]
    for i, row in enumerate(expanded):
        tag = "th" if i == 0 else "td"
        html_parts.append("<tr>" + "".join(f"<{tag}>{escape(c)}</{tag}>" for c in row) + "</tr>")
    html_parts.append("</table>")

    t["html"] = "".join(html_parts)
    t["rows"] = expanded[1:]
    t["headers"] = expanded[0]
    t["text"] = "\n".join(
        f"{a}: {b}" + (f" | {c}: {d}" if c else "")
        for a, b, c, d in expanded[1:]
        if a or c
    )
    t["source"] = t.get("source") or "ppstructure"
    t["cleaned"] = True
    return t


def _bbox_centers_overlap(a: List[Any], b: List[Any]) -> bool:
    """True if either bbox's center falls inside the other bbox."""
    if not a or not b or len(a) < 4 or len(b) < 4:
        return False

    def center_inside(inner: List[Any], outer: List[Any]) -> bool:
        cx = (inner[0] + inner[2]) / 2.0
        cy = (inner[1] + inner[3]) / 2.0
        return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

    return center_inside(a, b) or center_inside(b, a)


def enhance_page_forms(
    page: PageResult,
    image_rgb: Optional[np.ndarray] = None,
) -> PageResult:
    """Mutate/return page with better tables + checkbox fields."""
    cleaned_tables = []
    for t in page.tables or []:
        if isinstance(t, dict):
            cleaned_tables.append(clean_ppstructure_table(t))

    built_tables = rebuild_two_column_form_table(page)
    if built_tables:
        others = [
            t for t in cleaned_tables
            if (t.get("source") or "") not in ("two_column_form", "wide_grid")
        ]
        # A PPStructure table covering the same region as a geometric table we
        # just built is just a lower-quality re-derivation of the same fields,
        # not a distinct table - keeping it duplicates every field (once well
        # ordered from our own reconstruction, once poorly ordered from
        # PPStructure's own HTML cell order). Only keep it as a genuine
        # secondary table when it covers content we don't already cover.
        built_bboxes = [t.get("bbox") or [] for t in built_tables]
        distinct_others = [
            t for t in others
            if not any(_bbox_centers_overlap(t.get("bbox") or [], fb) for fb in built_bboxes)
        ]
        page.tables = built_tables + distinct_others[:1]
    else:
        page.tables = cleaned_tables

    page.checkboxes = detect_checkboxes(page, image_rgb)

    if page.checkboxes:
        text = page.text or ""
        extra = []
        for cb in page.checkboxes:
            label = cb.get("label") or "Checkbox"
            col = cb.get("column")
            suffix = f" ({col})" if col else ""
            if cb.get("source") == "standalone_indicator":
                # The bare label ("EOB Indicator") is already in the text at
                # its real position - fill its value in there rather than
                # appending a second copy of the field at the end of the page.
                pat = re.compile(r"(?<![\w:])" + re.escape(label) + r"(?![\w:])(?!\s*:)")
                text, n = pat.subn(f"{label}: {cb.get('value')}", text, count=1)
                if n:
                    continue
            extra.append(f"{label}{suffix}: {cb.get('value')}")
        page.text = text.rstrip() + ("\n" + "\n".join(extra) if extra else "")

    return page
