"""
Fast digital-PDF layout + table extraction (no OCR / no PPStructure).

Uses PyMuPDF text blocks + pdfplumber tables so formatted_text.txt gets
[TEXT] + [LAYOUT] + [TABLES] while staying seconds-fast.
"""

from __future__ import annotations

from html import escape
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber

from ocr.schemas import Block, Line, PageResult, Word


def _bbox_from_fitz(bbox: Any) -> List[int]:
    x0, y0, x1, y1 = bbox
    return [int(x0), int(y0), int(x1), int(y1)]


def _words_from_line(spans: List[Dict[str, Any]], line_bbox: List[int]) -> List[Word]:
    words: List[Word] = []
    for sp in spans:
        text = (sp.get("text") or "").strip()
        if not text:
            continue
        bb = sp.get("bbox")
        words.append(
            Word(
                text=text,
                conf=1.0,
                bbox=_bbox_from_fitz(bb) if bb else list(line_bbox),
            )
        )
    if not words and any((sp.get("text") or "").strip() for sp in spans):
        joined = " ".join((sp.get("text") or "").strip() for sp in spans).strip()
        if joined:
            words.append(Word(text=joined, conf=1.0, bbox=list(line_bbox)))
    return words


def _block_type_for(text: str, bbox: List[int], page_height: float) -> str:
    t = (text or "").strip()
    if not t:
        return "noise"
    # Header / footer bands
    if page_height > 0:
        if bbox[3] < page_height * 0.08:
            return "header"
        if bbox[1] > page_height * 0.92:
            return "footer"
    # Heading heuristic: short, mostly caps / title-like
    if len(t) <= 80 and (
        t.isupper()
        or t.startswith("SECTION ")
        or t.startswith("Section ")
        or t.endswith(":")
    ):
        return "heading"
    if ":" in t and len(t) <= 120:
        return "label_value"
    return "paragraph"


def _table_to_html(rows: List[List[Optional[str]]]) -> str:
    parts = ["<table>"]
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            val = escape((cell or "").strip())
            parts.append(f"<td>{val}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _table_to_text(rows: List[List[Optional[str]]]) -> str:
    lines = []
    for row in rows:
        cells = [(c or "").strip() for c in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _extract_tables_pdfplumber(page) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    try:
        found = page.find_tables() or []
    except Exception:
        found = []

    if found:
        for t in found:
            try:
                rows = t.extract() or []
            except Exception:
                rows = []
            if not rows:
                continue
            bbox = []
            try:
                bbox = [int(v) for v in t.bbox]
            except Exception:
                bbox = []
            html = _table_to_html(rows)
            tables.append(
                {
                    "bbox": bbox,
                    "html": html,
                    "text": _table_to_text(rows),
                    "confidence": 1.0,
                    "source": "pdfplumber",
                    "engine": "digital_pdf",
                    "cell_bbox": [],
                }
            )
        return tables

    # Fallback: extract_tables without bboxes
    try:
        raw = page.extract_tables() or []
    except Exception:
        raw = []
    for rows in raw:
        if not rows:
            continue
        tables.append(
            {
                "bbox": [],
                "html": _table_to_html(rows),
                "text": _table_to_text(rows),
                "confidence": 1.0,
                "source": "pdfplumber",
                "engine": "digital_pdf",
                "cell_bbox": [],
            }
        )
    return tables


def _blocks_and_lines_from_fitz(page: fitz.Page) -> Tuple[List[Block], List[Line], str]:
    data = page.get_text("dict")
    page_h = float(page.rect.height)
    blocks: List[Block] = []
    all_lines: List[Line] = []
    text_parts: List[str] = []
    order = 0

    for b in data.get("blocks") or []:
        if b.get("type") != 0:  # 0 = text
            continue
        line_objs: List[Line] = []
        block_text_parts: List[str] = []
        for li, ln in enumerate(b.get("lines") or []):
            spans = ln.get("spans") or []
            line_text = "".join(sp.get("text") or "" for sp in spans).strip()
            if not line_text:
                continue
            lb = _bbox_from_fitz(ln.get("bbox") or b.get("bbox") or (0, 0, 0, 0))
            words = _words_from_line(spans, lb)
            line_obj = Line(
                text=line_text,
                conf=1.0,
                bbox=lb,
                words=words,
                line_index=li,
            )
            line_objs.append(line_obj)
            all_lines.append(line_obj)
            block_text_parts.append(line_text)

        if not line_objs:
            continue

        bb = _bbox_from_fitz(b.get("bbox") or (0, 0, 0, 0))
        block_text = "\n".join(block_text_parts)
        btype = _block_type_for(block_text, bb, page_h)
        blocks.append(
            Block(
                block_type=btype,
                lines=line_objs,
                bbox=bb,
                reading_order=order,
            )
        )
        order += 1
        text_parts.append(block_text)

    # Reading order top-to-bottom, left-to-right
    blocks.sort(key=lambda x: (x.bbox[1], x.bbox[0]))
    for i, blk in enumerate(blocks):
        blk.reading_order = i

    page_text = "\n".join(text_parts).strip()
    return blocks, all_lines, page_text


def build_digital_page_result(
    fitz_page: fitz.Page,
    plumber_page: Optional[Any],
    page_index: int,
) -> PageResult:
    """
    Build one PageResult from a digital (native-text) PDF page using
    PyMuPDF for text/layout blocks and pdfplumber for table extraction.

    No rasterization and no OCR engine is involved - this is the fast path
    for pages that ocr.engine_router.route_pdf_pages classified as digital.
    """
    width = int(fitz_page.rect.width)
    height = int(fitz_page.rect.height)

    blocks, lines, block_text = _blocks_and_lines_from_fitz(fitz_page)
    plumber_text = (plumber_page.extract_text() or "").strip() if plumber_page is not None else ""
    page_text = plumber_text or block_text

    tables = _extract_tables_pdfplumber(plumber_page) if plumber_page is not None else []

    # Mark blocks that sit inside table bboxes as table_row
    for table in tables:
        tb = table.get("bbox") or []
        if len(tb) < 4:
            continue
        for blk in blocks:
            if not blk.bbox or len(blk.bbox) < 4:
                continue
            cx = (blk.bbox[0] + blk.bbox[2]) / 2.0
            cy = (blk.bbox[1] + blk.bbox[3]) / 2.0
            if tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]:
                if blk.block_type not in ("header", "footer", "noise"):
                    blk.block_type = "table_row"

    # Add explicit table blocks for structured output
    for ti, table in enumerate(tables):
        ttext = (table.get("text") or "").strip()
        if not ttext:
            continue
        tb = table.get("bbox") or [0, 0, width, height]
        tlines = [
            Line(
                text=ln,
                conf=1.0,
                bbox=list(tb),
                words=[Word(text=ln, conf=1.0, bbox=list(tb))],
                line_index=j,
            )
            for j, ln in enumerate(ttext.split("\n"))
            if ln.strip()
        ]
        if tlines:
            blocks.append(
                Block(
                    block_type="table",
                    lines=tlines,
                    bbox=list(tb),
                    reading_order=len(blocks) + ti,
                )
            )

    blocks.sort(key=lambda x: (x.bbox[1], x.bbox[0]))
    for j, blk in enumerate(blocks):
        blk.reading_order = j

    return PageResult(
        page_index=page_index,
        width=width,
        height=height,
        text=page_text,
        lines=lines,
        blocks=blocks,
        tables=tables,
    )


def build_digital_page_results(
    pdf_path: str,
    max_pages: int = 200,
) -> List[PageResult]:
    """
    Build PageResult list with text + layout blocks + tables from a digital PDF.

    Assumes every page in range is digital; for mixed digital/scanned PDFs,
    use ocr.hybrid_layout.route_pdf_document instead so scanned pages fall
    back to PaddleOCR individually.
    """
    results: List[PageResult] = []
    doc = fitz.open(pdf_path)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            n = min(len(doc), len(pdf.pages), max_pages)
            for i in range(n):
                results.append(
                    build_digital_page_result(doc.load_page(i), pdf.pages[i], i)
                )
    finally:
        doc.close()

    return results
