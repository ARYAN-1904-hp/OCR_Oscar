"""
Build regex-ready OCR text for a future standalone regex extractor.

Per page sections:
  [TEXT]    reading-order text (includes structured table grids)
  [LAYOUT]  blocks with type + bbox + text (PPStructure-aware)
  [TABLES]     complex table structure (rows/cols/cells from PPStructure HTML)
  [CHECKBOXES] Yes/No form checkbox states
  [BOXES]      line/word boxes
  [FIELDS]     emails / phones / NPI / tax IDs
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Tuple

from ocr.postprocess import fix_common_paddle_errors

_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?:\(\d{3}\)\s*|\d{3}[\s.-])\d{3}[\s.-]\d{4}")
_NPI = re.compile(r"\b\d{10}\b")
_TAX_ID = re.compile(r"\b\d{2}-\d{7}\b")
_LABEL_VALUE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z0-9 /().'_#-]{1,80}?)\s*:\s*(?P<value>.*)$"
)


class _TableGridExtractor(HTMLParser):
    """Parse HTML tables into a full cell grid (keeps empty cells for structure)."""

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
            # separate lines together with no boundary at all (e.g. a
            # two-sentence review-notes cell becoming "...complete.Approved
            # per..."). _clean()'s whitespace collapse below absorbs any
            # resulting double space where a run already ended in one.
            cell = _clean(" ".join(self._cell))
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


def _clean(text: str) -> str:
    t = (text or "").replace("\u00a0", " ").replace("..", ".")
    t = _WS.sub(" ", t).strip(" \t-|")
    # Same PaddleOCR-specific corrections applied to the plain-text output
    # (see ocr/postprocess.py) - table/layout cell text is built independently
    # of that path (it never goes through normalize_linebreaks), so without
    # this it silently misses fixes like a comma misread as a period or a
    # missing space between two sentences that a regex extractor downstream
    # would otherwise trip over.
    t = fix_common_paddle_errors(t)
    return t


def _bbox(obj: Dict[str, Any]) -> str:
    b = obj.get("bbox") or []
    if not b or len(b) < 4:
        return "[]"
    return f"[{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}]"


def _normalize_kv(text: str) -> str:
    t = _clean(text)
    if not t:
        return ""
    m = _LABEL_VALUE.match(t)
    if m:
        label = _clean(m.group("label"))
        value = _clean(m.group("value"))
        return f"{label}: {value}" if value else f"{label}:"
    return t


def _table_html_to_grid(table_html: str) -> List[List[str]]:
    if not table_html:
        return []
    parser = _TableGridExtractor()
    try:
        parser.feed(table_html)
    except Exception:
        return []
    return parser.rows


def _pipe_text_to_grid(text: str) -> List[List[str]]:
    grid: List[List[str]] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or "|" not in s:
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if not cells:
            continue
        if all(set(c.replace(":", "")) <= {"-", ""} for c in cells):
            continue
        grid.append(cells)
    return grid


def _table_to_grid(table: Dict[str, Any]) -> List[List[str]]:
    html = table.get("html") or ""
    grid = _table_html_to_grid(html)
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
    return _pipe_text_to_grid(table.get("text") or "")


def _grid_stats(grid: List[List[str]]) -> Tuple[int, int]:
    rows = len(grid)
    cols = max((len(r) for r in grid), default=0)
    return rows, cols


def _format_table_grid_lines(grid: List[List[str]], indent: str = "  ") -> List[str]:
    """Emit structured row/cell lines for complex tables."""
    if not grid:
        return []
    rows, cols = _grid_stats(grid)
    out: List[str] = [f"{indent}shape: {rows}x{cols}"]
    norm = [list(r) + [""] * (cols - len(r)) for r in grid]
    if norm:
        header = " | ".join(c if c else "-" for c in norm[0])
        out.append(f"{indent}HEADER | {header}")
    # Skip row 0 — already emitted as HEADER
    for ri, row in enumerate(norm[1:], start=1):
        cells = [c if c else "-" for c in row]
        out.append(f"{indent}ROW {ri} | " + " | ".join(cells))
        if cols >= 2:
            for ci, cell in enumerate(row):
                if cell:
                    out.append(f"{indent}  CELL[{ri},{ci}] = {cell}")
    return out


def _table_to_text_block(table: Dict[str, Any], index: int) -> List[str]:
    """Compact structured table for inclusion inside [TEXT]."""
    source = table.get("source") or table.get("engine") or "layout"
    lines = [f"[TABLE {index} | source={source}]"]
    grid = _table_to_grid(table)
    if grid:
        lines.extend(_format_table_grid_lines(grid, indent=""))
    else:
        text = _clean(table.get("text") or "")
        if text:
            for ln in text.split("\n"):
                cleaned = _normalize_kv(ln)
                if cleaned:
                    lines.append(cleaned)
    lines.append(f"[/TABLE {index}]")
    return lines


def _y_of(obj: Dict[str, Any]) -> int:
    b = obj.get("bbox") or []
    return int(b[1]) if len(b) >= 2 else 0


def _x_of(obj: Dict[str, Any]) -> int:
    b = obj.get("bbox") or []
    return int(b[0]) if len(b) >= 1 else 0


def _h_overlap_frac(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = min(a1, b1) - max(a0, b0)
    width = min(a1 - a0, b1 - b0)
    if width <= 0:
        return 0.0
    return max(0.0, inter) / width


def _bare_label_and_value(text: str) -> Tuple[str, str]:
    """Split 'Label:' / 'Label: value' into (label, value) via _LABEL_VALUE."""
    m = _LABEL_VALUE.match((text or "").strip())
    if not m:
        return "", ""
    return _clean(m.group("label")), _clean(m.group("value"))


def _flatten_lines_with_bbox(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One entry per OCR line across all blocks, keeping its own bbox and
    (block_idx, line_idx) so it can be looked up and consumed later."""
    out: List[Dict[str, Any]] = []
    for bi, block in enumerate(page.get("blocks") or []):
        btype = (block.get("block_type") or "").lower()
        for li, ln in enumerate(block.get("lines") or []):
            bbox = ln.get("bbox") or []
            if len(bbox) < 4:
                continue
            out.append(
                {
                    "text": ln.get("text") or "",
                    "bbox": bbox,
                    "block_type": btype,
                    "key": (bi, li),
                }
            )
    return out


def _merge_stacked_label_values(
    page: Dict[str, Any],
) -> Tuple[Dict[Tuple[int, int], str], set]:
    """
    Row-by-row text ordering keeps every label next to other labels in its
    own visual row/band, and every value next to other values in the row
    below - so a label whose value sits UNDER it (rather than beside it)
    ends up textually far from that value, often with unrelated fields
    sandwiched in between.

    This pairs each orphan label (a 'Label:' line with nothing after the
    colon) with the nearest line below it - by bbox, not stream position -
    that shares enough horizontal overlap and isn't itself a bare label.
    Returns:
      overrides: {(block_idx, line_idx): "Label: value"} for label lines
      consumed:  {(block_idx, line_idx)} for value lines already merged in,
                 so the renderer can skip them at their original position.
    """
    entries = _flatten_lines_with_bbox(page)
    consumed: set = set()
    overrides: Dict[Tuple[int, int], str] = {}

    for entry in entries:
        if entry["key"] in consumed:
            continue
        label, value = _bare_label_and_value(entry["text"])
        if not label or value:
            continue  # not a label line, or value already on the same line

        lx0, ly0, lx1, ly1 = entry["bbox"]
        label_h = max(ly1 - ly0, 1)
        max_gap = max(3 * label_h, 60)

        best_key = None
        best_gap = None
        for cand in entries:
            if cand["key"] == entry["key"] or cand["key"] in consumed:
                continue
            if cand["block_type"] in ("heading", "table_row", "noise"):
                continue
            if not cand["text"].strip():
                continue
            clabel, cvalue = _bare_label_and_value(cand["text"])
            if clabel and not cvalue:
                continue  # candidate is itself an orphan label, not a value

            cx0, cy0, cx1, cy1 = cand["bbox"]
            if cy0 < ly1 - 3:
                continue  # not below the label
            if _h_overlap_frac(lx0, lx1, cx0, cx1) < 0.3:
                continue
            gap = cy0 - ly1
            if gap > max_gap:
                continue
            if best_gap is None or gap < best_gap:
                best_key, best_gap = cand["key"], gap

        if best_key is not None:
            value_text = next(c["text"] for c in entries if c["key"] == best_key)
            merged_value = _clean(value_text)
            overrides[entry["key"]] = (
                f"{label}: {merged_value}" if merged_value else f"{label}:"
            )
            consumed.add(best_key)

    return overrides, consumed


def _center_in_table(block: Dict[str, Any], table: Dict[str, Any]) -> bool:
    bb = block.get("bbox") or []
    tb = table.get("bbox") or []
    if len(bb) < 4 or len(tb) < 4:
        return False
    cx = (bb[0] + bb[2]) / 2.0
    cy = (bb[1] + bb[3]) / 2.0
    return tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]


def _page_text_section(page: Dict[str, Any]) -> List[str]:
    """
    Reading-order text + tables (as in the page image).
    Tables are inserted at their vertical position; overlapping table_row
    blocks are skipped to avoid duplicate garbage.
    """
    lines: List[str] = ["[TEXT]"]
    blocks = list(page.get("blocks") or [])
    tables = [t for t in (page.get("tables") or []) if isinstance(t, dict)]
    overrides, consumed = _merge_stacked_label_values(page)

    events: List[Tuple[int, int, str, Any]] = []
    for i, table in enumerate(tables):
        events.append((_y_of(table), _x_of(table), "table", i))

    for bi, block in enumerate(blocks):
        btype = (block.get("block_type") or "").lower()
        if btype == "noise":
            continue
        # Skip content covered by a detected table (table carries structure)
        if any(_center_in_table(block, t) for t in tables):
            continue
        if btype in ("table", "table_row"):
            continue
        events.append((_y_of(block), _x_of(block), "block", (bi, block)))

    events.sort(key=lambda e: (e[0], e[1]))

    emitted_tables = set()
    if events:
        for _, __, kind, payload in events:
            if kind == "table":
                idx = int(payload)
                if idx in emitted_tables:
                    continue
                emitted_tables.add(idx)
                lines.append("")
                lines.extend(_table_to_text_block(tables[idx], idx))
                lines.append("")
            else:
                bi, block = payload
                for li, ln in enumerate(block.get("lines") or []):
                    key = (bi, li)
                    if key in consumed:
                        continue  # merged into an orphan label above as its value
                    raw_text = overrides.get(key, ln.get("text") or "")
                    cleaned = _normalize_kv(raw_text)
                    if cleaned:
                        lines.append(cleaned)
    else:
        for ln in (page.get("text") or "").split("\n"):
            cleaned = _normalize_kv(ln)
            if cleaned:
                lines.append(cleaned)

    # Any tables not placed (no bbox) — append at end
    for i, table in enumerate(tables):
        if i not in emitted_tables:
            lines.append("")
            lines.extend(_table_to_text_block(table, i))

    lines.append("[/TEXT]")
    return lines


def _page_layout_section(page: Dict[str, Any]) -> List[str]:
    blocks = page.get("blocks") or []
    if not blocks:
        return []
    lines: List[str] = ["[LAYOUT]"]
    for i, block in enumerate(blocks):
        btype = (block.get("block_type") or "unknown").lower()
        if btype == "noise":
            continue
        text = _clean(block.get("text") or "")
        if not text and btype != "table":
            continue
        lines.append(
            f"BLOCK {i} | type={btype} | bbox={_bbox(block)} | order={block.get('reading_order', i)}"
        )
        for ln in (text or "").split("\n"):
            cleaned = _normalize_kv(ln)
            if cleaned:
                lines.append(f"  {cleaned}")
    lines.append("[/LAYOUT]")
    return lines


def _page_tables_section(page: Dict[str, Any]) -> List[str]:
    tables = page.get("tables") or []
    if not tables:
        return []
    lines: List[str] = ["[TABLES]"]
    for i, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        source = table.get("source") or table.get("engine") or "layout"
        html = table.get("html") or ""
        grid = _table_to_grid(table)
        rows, cols = _grid_stats(grid)
        conf = table.get("confidence")
        conf_s = f"{float(conf):.3f}" if conf is not None else "-"
        lines.append(
            f"TABLE {i} | source={source} | bbox={_bbox(table)} "
            f"| shape={rows}x{cols} | conf={conf_s}"
        )
        if grid:
            lines.extend(_format_table_grid_lines(grid, indent="  "))
        else:
            text = _clean(table.get("text") or "")
            if text:
                for ln in text.split("\n"):
                    cleaned = _normalize_kv(ln)
                    if cleaned:
                        lines.append(f"  {cleaned}")
        # Keep raw HTML snippet marker for downstream tools (truncated)
        if html and source == "ppstructure":
            compact = _WS.sub(" ", html).strip()
            if len(compact) > 400:
                compact = compact[:400] + "…"
            lines.append(f"  HTML: {compact}")
    lines.append("[/TABLES]")
    return lines


def _page_boxes_section(page: Dict[str, Any]) -> List[str]:
    page_lines = page.get("lines") or []
    if not page_lines:
        return []
    lines: List[str] = ["[BOXES]"]
    for i, ln in enumerate(page_lines):
        text = _clean(ln.get("text") or "")
        if not text:
            continue
        conf = ln.get("conf")
        conf_s = f"{float(conf):.3f}" if conf is not None else "-"
        lines.append(
            f"LINE {i} | bbox={_bbox(ln)} | conf={conf_s} | text={_normalize_kv(text)}"
        )
        words = ln.get("words") or []
        if len(words) >= 2:
            for w in words:
                wt = _clean(w.get("text") or "")
                if not wt:
                    continue
                lines.append(f"  WORD | bbox={_bbox(w)} | text={wt}")
    lines.append("[/BOXES]")
    return lines


def _page_checkboxes_section(page: Dict[str, Any]) -> List[str]:
    boxes = page.get("checkboxes") or []
    if not boxes:
        return []
    lines: List[str] = ["[CHECKBOXES]"]
    for i, cb in enumerate(boxes):
        if not isinstance(cb, dict):
            continue
        label = _clean(cb.get("label") or "Checkbox")
        value = _clean(str(cb.get("value") or ""))
        col = cb.get("column") or ""
        src = cb.get("source") or ""
        cy = cb.get("checked_yes")
        cn = cb.get("checked_no")
        extra = []
        if col:
            extra.append(f"column={col}")
        if src:
            extra.append(f"source={src}")
        if cy is not None:
            extra.append(f"yes_ink={cy}")
        if cn is not None:
            extra.append(f"no_ink={cn}")
        meta = (" | " + " | ".join(extra)) if extra else ""
        lines.append(f"CHECKBOX {i} | {label}: {value}{meta}")
    lines.append("[/CHECKBOXES]")
    return lines


def _page_fields_section(page: Dict[str, Any]) -> List[str]:
    text = page.get("text") or ""
    # Also scan table plain text
    for table in page.get("tables") or []:
        if isinstance(table, dict):
            text += "\n" + (table.get("text") or "")
    extras: List[str] = []
    for email in _EMAIL.findall(text):
        extras.append(f"Email: {email}")
    for phone in _PHONE.findall(text):
        extras.append(f"Phone: {phone}")
    for npi in _NPI.findall(text):
        extras.append(f"NPI: {npi}")
    for tax in _TAX_ID.findall(text):
        extras.append(f"Federal Tax ID: {tax}")
    if not extras:
        return []
    seen = set()
    out = ["[FIELDS]"]
    for line in extras:
        if line not in seen:
            seen.add(line)
            out.append(line)
    out.append("[/FIELDS]")
    return out


def build_regex_ready_text(ocr_result: Dict[str, Any]) -> str:
    """Combined text: TEXT + LAYOUT + TABLES + CHECKBOXES + BOXES + FIELDS."""
    pages = ocr_result.get("pages") or []
    out: List[str] = []

    meta = ocr_result.get("meta") or {}
    if meta:
        engine = meta.get("engine_used") or meta.get("engine") or ""
        layout_engine = meta.get("layout_engine") or ""
        out.append(
            f"# OCR meta | engine={engine} | layout={layout_engine} "
            f"| visual={meta.get('layout_visual_ocr', False)}"
        )
        out.append("")

    if not pages:
        body = _clean(ocr_result.get("text") or "")
        return (body + "\n") if body else ""

    for page in pages:
        idx = int(page.get("page_index") or 0) + 1
        width = int(page.get("width") or 0)
        height = int(page.get("height") or 0)
        n_blocks = len(page.get("blocks") or [])
        n_tables = len(page.get("tables") or [])
        n_cb = len(page.get("checkboxes") or [])
        out.append(f"=== PAGE {idx} ===")
        if width and height:
            out.append(
                f"size: {width}x{height} | blocks={n_blocks} | tables={n_tables} | checkboxes={n_cb}"
            )
        else:
            out.append(f"blocks={n_blocks} | tables={n_tables} | checkboxes={n_cb}")
        out.append("")

        out.extend(_page_text_section(page))
        out.append("")
        layout = _page_layout_section(page)
        if layout:
            out.extend(layout)
            out.append("")
        tables = _page_tables_section(page)
        if tables:
            out.extend(tables)
            out.append("")
        checkboxes = _page_checkboxes_section(page)
        if checkboxes:
            out.extend(checkboxes)
            out.append("")
        boxes = _page_boxes_section(page)
        if boxes:
            out.extend(boxes)
            out.append("")
        fields = _page_fields_section(page)
        if fields:
            out.extend(fields)
            out.append("")

    text = "\n".join(out)
    text = _MULTI_NL.sub("\n\n", text).strip() + "\n"
    return text
