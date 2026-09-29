"""Hybrid layout: heuristic baseline + PPStructure table/layout merge."""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
import numpy as np

from ocr.schemas import Block, Column, Line, PageResult, Word
from ocr.layout import analyse_layout
from ocr.layout_adapter import LayoutAdapter, LayoutResult
from ocr.engine_router import ROUTE_DIGITAL, route_pdf_pages
from ocr.digital_layout import build_digital_page_result
from ocr.pdf_utils import render_pdf_page_to_image

# PPStructure is slow (extra models per page). Off by default for feasible speed.
# Enable only when needed: set OCR_USE_PPSTRUCTURE=1
_USE_PPSTRUCTURE = os.environ.get("OCR_USE_PPSTRUCTURE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

_TAG_RE = re.compile(r"<[^>]+>")


class _HtmlToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self.parts.append(data.strip())


def _html_to_plain(html: str) -> str:
    if not html:
        return ""
    if "<" not in html:
        return html.strip()
    parser = _HtmlToText()
    try:
        parser.feed(html)
        return "\n".join(parser.parts)
    except Exception:
        return _TAG_RE.sub(" ", html).strip()


def _as_bgr(image: Any) -> Any:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr[:, :, ::-1].copy()
    return arr


def _center_inside(bbox: List[int], region: List[int]) -> bool:
    if not bbox or not region or len(bbox) < 4 or len(region) < 4:
        return False
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _lines_from_text(text: str, bbox: List[int], conf: float) -> List[Line]:
    lines: List[Line] = []
    for i, raw in enumerate((text or "").split("\n")):
        t = raw.strip()
        if not t:
            continue
        lines.append(
            Line(
                text=t,
                conf=float(conf or 0.0),
                bbox=list(bbox) if bbox else [0, 0, 0, 0],
                words=[Word(text=t, conf=float(conf or 0.0), bbox=list(bbox) if bbox else [0, 0, 0, 0])],
                line_index=i,
            )
        )
    return lines


def _normalize_table_dict(table: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure table has html/text/source for downstream formatted_text."""
    from html import escape as _esc

    t = dict(table)
    t.setdefault("source", t.get("engine") or "layout")
    if not t.get("html"):
        headers = t.get("headers") or []
        rows = t.get("rows") or []
        if headers or rows:
            parts = ["<table>"]
            if headers:
                parts.append(
                    "<tr>"
                    + "".join(f"<th>{_esc(str(c))}</th>" for c in headers)
                    + "</tr>"
                )
            for r in rows:
                parts.append(
                    "<tr>"
                    + "".join(f"<td>{_esc(str(c))}</td>" for c in (r or []))
                    + "</tr>"
                )
            parts.append("</table>")
            t["html"] = "".join(parts)
        elif t.get("text") and "|" in (t.get("text") or ""):
            # Markdown / pipe table → simple HTML
            grid_rows = []
            for line in (t.get("text") or "").splitlines():
                s = line.strip()
                if not s or set(s.replace("|", "").replace(":", "").replace("-", "").strip()) <= {""}:
                    continue
                if "|" in s:
                    cells = [c.strip() for c in s.strip("|").split("|")]
                    if cells and not all(set(c) <= {"-", ":"} for c in cells):
                        grid_rows.append(cells)
            if grid_rows:
                parts = ["<table>"]
                for i, r in enumerate(grid_rows):
                    tag = "th" if i == 0 else "td"
                    parts.append(
                        "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in r) + "</tr>"
                    )
                parts.append("</table>")
                t["html"] = "".join(parts)
    if not t.get("text") and t.get("html"):
        t["text"] = _html_to_plain(t["html"])
    return t


class HybridLayoutEngine:
    def __init__(self):
        self.pp_engine = None
        self.adapter = None
        self.ppstructure_enabled = bool(_USE_PPSTRUCTURE)

    def _ensure_pp(self) -> bool:
        if self.pp_engine is not None and self.adapter is not None:
            return self.pp_engine.is_available()
        try:
            from ocr.ppstructure_engine import PPStructureEngine

            self.pp_engine = PPStructureEngine()
            self.adapter = LayoutAdapter()
            self.ppstructure_enabled = self.pp_engine.is_available()
            return self.ppstructure_enabled
        except Exception as e:
            print(f"[HybridLayout] PPStructure init failed: {e}")
            self.ppstructure_enabled = False
            return False

    def analyse(
        self,
        words: List[Dict],
        page_width: int,
        page_height: int,
        image: Optional[Any] = None,
        use_ppstructure: Optional[bool] = None,
    ) -> Tuple[List[Block], List[Column], List[Dict[str, Any]]]:
        blocks, columns, tables = analyse_layout(
            words, page_width=page_width, page_height=page_height
        )
        tables = [_normalize_table_dict(t) for t in (tables or []) if isinstance(t, dict)]

        want_pp = _USE_PPSTRUCTURE if use_ppstructure is None else bool(use_ppstructure)
        if image is None or not want_pp:
            return blocks, columns, tables
        if not self._ensure_pp():
            return blocks, columns, tables

        try:
            pp_result = self.pp_engine.analyse(_as_bgr(image), words=words)
        except Exception as e:
            print(f"[HybridLayout] PPStructure failed: {e}")
            return blocks, columns, tables

        if pp_result is None:
            return blocks, columns, tables

        try:
            pp_layout = self.adapter.from_ppstructure(pp_result)
        except Exception as e:
            print(f"[HybridLayout] LayoutAdapter failed: {e}")
            return blocks, columns, tables

        return self._merge(blocks, columns, tables, pp_layout)

    def _merge(
        self,
        blocks: List[Block],
        columns: List[Column],
        tables: List[Dict[str, Any]],
        pp_layout: LayoutResult,
    ) -> Tuple[List[Block], List[Column], List[Dict[str, Any]]]:
        table_regions = [
            list(t.bbox) for t in pp_layout.tables if t.bbox and len(t.bbox) >= 4
        ]

        # Tag heuristic blocks that sit inside PPStructure table regions
        for block in blocks:
            if block.block_type in ("noise", "header", "footer"):
                continue
            for region in table_regions:
                if _center_inside(block.bbox, region):
                    if block.block_type not in ("table", "table_row"):
                        block.block_type = "table_row"
                    break

        # Prefer PPStructure tables (complex HTML structure); keep heuristic as fallback
        pp_tables: List[Dict[str, Any]] = []
        for t in pp_layout.tables:
            html = (t.metadata.get("html") or "").strip()
            plain = _html_to_plain(html) if html else (t.text or "").strip()
            if not html and not plain:
                continue
            pp_tables.append(
                {
                    "bbox": list(t.bbox) if t.bbox else [],
                    "html": html or t.text,
                    "text": plain,
                    "confidence": float(t.confidence or 0.0),
                    "source": "ppstructure",
                    "cell_bbox": t.metadata.get("cell_bbox", []),
                    "engine": "ppstructure",
                }
            )

        merged_tables = pp_tables if pp_tables else list(tables or [])
        for ht in tables or []:
            if not isinstance(ht, dict):
                continue
            # Keep heuristic tables that don't overlap a PPStructure table
            hb = ht.get("bbox") or []
            overlaps = False
            for pt in pp_tables:
                if _center_inside(hb, pt.get("bbox") or []):
                    overlaps = True
                    break
            if not overlaps and (ht.get("html") or ht.get("text")):
                item = dict(ht)
                item.setdefault("source", "heuristic")
                merged_tables.append(item)

        # Inject PPStructure non-table regions as layout blocks (title/text/figure)
        # so formatted_text [LAYOUT] clearly shows PPStructure structure.
        n_existing = len(blocks)
        next_order = (max((b.reading_order for b in blocks), default=-1) + 1)
        for region in pp_layout.blocks:
            rtype = (region.block_type or "text").lower()
            if rtype == "table":
                # Heuristic blocks inside this same region were already
                # re-tagged "table_row"/"table" above (via table_regions) and
                # carry the geometry-derived reading order, which is more
                # reliable than PPStructure's raw HTML cell order for
                # label/value forms. Injecting the HTML-derived text here too
                # would duplicate every field (once from the heuristic block,
                # once from this block) in page.text. Only add this block
                # when it covers area the heuristic pass didn't already emit.
                region_bbox = list(region.bbox or [])
                already_covered = any(
                    (
                        _center_inside(region_bbox, b.bbox)
                        or _center_inside(b.bbox, region_bbox)
                    )
                    for b in blocks
                    if b.block_type != "noise"
                )
                if already_covered:
                    continue

                # Represent table as a dedicated block with plain cell text
                plain = _html_to_plain(region.metadata.get("html") or region.text or "")
                if not plain:
                    continue
                lines = _lines_from_text(plain, region_bbox, region.confidence)
                if not lines:
                    continue
                blocks.append(
                    Block(
                        block_type="table",
                        lines=lines,
                        bbox=region_bbox or [0, 0, 0, 0],
                        reading_order=next_order,
                    )
                )
                next_order += 1
                continue

            text = (region.text or "").strip()
            if not text and rtype in ("title", "header"):
                text = (region.metadata.get("text") or "").strip()
            if not text:
                continue

            # Skip if a heuristic block already covers the same center
            if any(
                _center_inside(list(region.bbox or []), b.bbox)
                or _center_inside(b.bbox, list(region.bbox or []))
                for b in blocks
                if b.block_type != "noise"
            ):
                # Still upgrade type when PPStructure is more specific
                for b in blocks:
                    if b.block_type == "noise":
                        continue
                    if _center_inside(b.bbox, list(region.bbox or [])) or _center_inside(
                        list(region.bbox or []), b.bbox
                    ):
                        if rtype in ("title", "header") and b.block_type == "paragraph":
                            b.block_type = "heading" if rtype == "title" else "header"
                        break
                continue

            mapped = {
                "title": "heading",
                "header": "header",
                "figure": "caption",
                "image": "caption",
                "chart": "caption",
                "text": "paragraph",
            }.get(rtype, "paragraph")

            lines = _lines_from_text(text, list(region.bbox or []), region.confidence)
            if not lines:
                continue
            blocks.append(
                Block(
                    block_type=mapped,
                    lines=lines,
                    bbox=list(region.bbox) if region.bbox else [0, 0, 0, 0],
                    reading_order=next_order,
                )
            )
            next_order += 1

        # `blocks[:n_existing]` already carries the correct column-major
        # reading order that analyse_layout() (and its column detection)
        # established - re-sorting the whole list from scratch by (y, x)
        # here would undo that and re-interleave side-by-side blocks whose
        # bboxes start at slightly different y. Only the newly-injected
        # PPStructure blocks (appended above, in whatever order PPStructure
        # returned them) need placing; stable-merge them into the existing
        # sequence by y-position instead of re-deriving reading order for
        # everything.
        existing_blocks = blocks[:n_existing]
        new_blocks = sorted(
            blocks[n_existing:],
            key=lambda b: b.bbox[1] if b.bbox and len(b.bbox) >= 4 else 0,
        )
        merged: List[Block] = []
        i = j = 0
        while i < len(existing_blocks) or j < len(new_blocks):
            if j >= len(new_blocks):
                merged.append(existing_blocks[i]); i += 1
            elif i >= len(existing_blocks):
                merged.append(new_blocks[j]); j += 1
            else:
                ey = existing_blocks[i].bbox[1] if existing_blocks[i].bbox else 0
                ny = new_blocks[j].bbox[1] if new_blocks[j].bbox else 0
                if ey <= ny:
                    merged.append(existing_blocks[i]); i += 1
                else:
                    merged.append(new_blocks[j]); j += 1
        for idx, b in enumerate(merged):
            b.reading_order = idx
        blocks = merged

        print(
            "[HybridLayout] Merged PPStructure layout "
            f"(regions={len(pp_layout.blocks)}, tables={len(pp_layout.tables)}, "
            f"out_tables={len(merged_tables)}, out_blocks={len(blocks)})."
        )
        return blocks, columns, merged_tables


# --------------------------------------------------------------------------
# Digital-first routing across a whole PDF
#
# Each page is classified up front (ocr.engine_router.route_pdf_pages) by
# inspecting its native text layer - no rasterization yet. Digital pages
# are extracted directly (PyMuPDF + pdfplumber, no OCR engine involved).
# Only pages that come back "scanned" are rasterized - one page at a time -
# and handed to the caller's `ocr_page_fn`, which is expected to run
# PaddleOCR + HybridLayoutEngine.analyse and return a PageResult. Deep OCR
# is never invoked for a page that already has a usable text layer.
# --------------------------------------------------------------------------

def route_pdf_document(
    pdf_path: str,
    ocr_page_fn: Callable[[np.ndarray, int], Tuple[PageResult, Dict[str, Any]]],
    max_pages: Optional[int] = None,
    dpi: int = 200,
    min_chars: Optional[int] = None,
    min_alpha_chars: Optional[int] = None,
    image_coverage_threshold: Optional[float] = None,
) -> Tuple[List[PageResult], List[Dict[str, Any]]]:
    """
    Route every page of a PDF to the cheapest engine that can read it.

    Returns (page_results, debug_info), both ordered by page_index. Each
    debug_info entry always carries "page_index", "route" ("digital" or
    "scanned"), and "reason"; scanned-page entries are further enriched
    with whatever `ocr_page_fn` returns as its debug dict.
    """
    route_kwargs: Dict[str, Any] = {}
    if min_chars is not None:
        route_kwargs["min_chars"] = min_chars
    if min_alpha_chars is not None:
        route_kwargs["min_alpha_chars"] = min_alpha_chars
    if image_coverage_threshold is not None:
        route_kwargs["image_coverage_threshold"] = image_coverage_threshold

    routes = route_pdf_pages(pdf_path, max_pages=max_pages, **route_kwargs)

    page_results: List[PageResult] = []
    debug_info: List[Dict[str, Any]] = []

    doc = fitz.open(pdf_path)
    try:
        with pdfplumber.open(pdf_path) as plumber_doc:
            n_plumber = len(plumber_doc.pages)
            for route in routes:
                idx = route.page_index
                fitz_page = doc.load_page(idx)

                if route.route == ROUTE_DIGITAL:
                    plumber_page = plumber_doc.pages[idx] if idx < n_plumber else None
                    pr = build_digital_page_result(fitz_page, plumber_page, idx)
                    page_results.append(pr)
                    debug_info.append(
                        {
                            "page_index": idx,
                            "route": route.route,
                            "reason": route.reason,
                            "engine_used": "digital_pdf",
                            "char_count": route.char_count,
                        }
                    )
                    continue

                img_rgb = render_pdf_page_to_image(fitz_page, dpi=dpi)
                pr, ocr_debug = ocr_page_fn(img_rgb, idx)
                page_results.append(pr)
                entry: Dict[str, Any] = {
                    "page_index": idx,
                    "route": route.route,
                    "reason": route.reason,
                    "char_count": route.char_count,
                }
                if isinstance(ocr_debug, dict):
                    entry.update(ocr_debug)
                debug_info.append(entry)
    finally:
        doc.close()

    return page_results, debug_info
