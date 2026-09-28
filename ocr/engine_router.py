from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np

# --------------------------------------------------------------------------
# Digital-first routing
#
# Decide, per PDF page and *before any rasterization*, whether the page has
# a native/selectable text layer ("digital") or must be rasterized and run
# through PaddleOCR ("scanned"). This lets mixed PDFs (e.g. a digital cover
# page followed by scanned exhibits) route each page to the cheapest engine
# capable of reading it, instead of an all-or-nothing document-level guess.
# --------------------------------------------------------------------------

ROUTE_DIGITAL = "digital"
ROUTE_SCANNED = "scanned"

# Tunable via env so ops can adjust sensitivity without a code change.
_DEFAULT_MIN_CHARS = int(os.environ.get("OCR_DIGITAL_MIN_CHARS", "40"))
_DEFAULT_MIN_ALPHA_CHARS = int(os.environ.get("OCR_DIGITAL_MIN_ALPHA_CHARS", "20"))
_DEFAULT_IMAGE_COVERAGE_THRESHOLD = float(
    os.environ.get("OCR_DIGITAL_IMAGE_COVERAGE_THRESHOLD", "0.85")
)


@dataclass
class PageRoute:
    page_index: int
    route: str              # ROUTE_DIGITAL | ROUTE_SCANNED
    char_count: int         # total stripped chars found in the native text layer
    alpha_char_count: int   # alnum-only subset (filters whitespace/punctuation noise)
    image_coverage: float   # fraction of page area covered by embedded raster images
    reason: str

    @property
    def is_digital(self) -> bool:
        return self.route == ROUTE_DIGITAL


def _alpha_char_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


def _image_coverage_ratio(page: "fitz.Page") -> float:
    """Fraction of the page covered by embedded raster images (0..1)."""
    try:
        page_area = float(page.rect.width * page.rect.height)
    except Exception:
        return 0.0
    if page_area <= 0:
        return 0.0

    covered = 0.0
    try:
        for img in page.get_images(full=True):
            xref = img[0]
            for rect in page.get_image_rects(xref):
                covered += float(rect.width * rect.height)
    except Exception:
        return 0.0
    return min(1.0, covered / page_area)


def inspect_pdf_page(
    page: "fitz.Page",
    min_chars: int = _DEFAULT_MIN_CHARS,
    min_alpha_chars: int = _DEFAULT_MIN_ALPHA_CHARS,
    image_coverage_threshold: float = _DEFAULT_IMAGE_COVERAGE_THRESHOLD,
) -> PageRoute:
    """
    Classify a single PDF page as digital or scanned by inspecting its
    native text layer with PyMuPDF. This never rasterizes the page -
    `page.get_text()` reads the PDF's text objects directly.
    """
    try:
        text = (page.get_text("text") or "").strip()
    except Exception:
        text = ""

    char_count = len(text)
    alpha_count = _alpha_char_count(text)
    coverage = _image_coverage_ratio(page)

    has_text_layer = char_count >= min_chars and alpha_count >= min_alpha_chars
    if has_text_layer:
        # Guard against a near full-bleed scanned image that also happens to
        # carry a thin sliver of real text (stamps, watermarks, page
        # numbers) - the page content itself still needs OCR.
        if coverage >= image_coverage_threshold and char_count < min_chars * 2:
            return PageRoute(
                page.number,
                ROUTE_SCANNED,
                char_count,
                alpha_count,
                coverage,
                "text layer too sparse relative to full-page image coverage",
            )
        return PageRoute(
            page.number, ROUTE_DIGITAL, char_count, alpha_count, coverage,
            "native text layer present",
        )

    return PageRoute(
        page.number, ROUTE_SCANNED, char_count, alpha_count, coverage,
        "no usable native text layer",
    )


def route_pdf_pages(
    pdf_path: str,
    max_pages: Optional[int] = None,
    min_chars: int = _DEFAULT_MIN_CHARS,
    min_alpha_chars: int = _DEFAULT_MIN_ALPHA_CHARS,
    image_coverage_threshold: float = _DEFAULT_IMAGE_COVERAGE_THRESHOLD,
) -> List[PageRoute]:
    """
    Classify every page of a PDF as digital or scanned, before any page is
    rasterized. Downstream code uses this to bypass PaddleOCR entirely for
    digital pages and only rasterize+OCR the scanned ones.
    """
    routes: List[PageRoute] = []
    doc = fitz.open(pdf_path)
    try:
        n = len(doc) if max_pages is None else min(len(doc), max_pages)
        for i in range(n):
            routes.append(
                inspect_pdf_page(
                    doc.load_page(i),
                    min_chars=min_chars,
                    min_alpha_chars=min_alpha_chars,
                    image_coverage_threshold=image_coverage_threshold,
                )
            )
    finally:
        doc.close()
    return routes


def score_words(words: List[Dict]) -> float:
    """Score OCR word list (higher is better). Favors confidence and text volume."""
    if not words:
        return 0.0
    total_chars = sum(len((w.get("text") or "").strip()) for w in words)
    avg_conf = sum(float(w.get("conf") or 0.0) for w in words) / max(1, len(words))
    return float(avg_conf * (1.0 + min(total_chars / 200.0, 5.0)))


def _avg_conf(words: List[Dict]) -> float:
    if not words:
        return 0.0
    return float(sum(float(w.get("conf") or 0.0) for w in words) / max(1, len(words)))


def _conf_above_ratio(words: List[Dict], threshold: float) -> float:
    """Fraction of words with confidence >= threshold (0..1)."""
    if not words:
        return 0.0
    n = sum(1 for w in words if float(w.get("conf") or 0.0) >= threshold)
    return n / len(words)


def _word_heights(words: List[Dict]) -> List[float]:
    out: List[float] = []
    for w in words:
        b = w.get("bbox")
        if b and len(b) >= 4:
            out.append(float(b[3] - b[1]))
    return out


def _coefficient_of_variation(values: List[float]) -> float:
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return (variance ** 0.5) / mean


def _likely_handwritten(img_rgb: np.ndarray, paddle_words: List[Dict]) -> bool:
    """
    Heuristic: is this image likely handwritten vs. printed?
    Used for logging/debug only - PaddleOCR handles both cases now.
    """
    if img_rgb is None or not paddle_words:
        return False

    total_chars = sum(len((w.get("text") or "").strip()) for w in paddle_words)
    confs = [float(w.get("conf") or 0.0) for w in paddle_words]
    avg_conf = sum(confs) / len(confs)

    if total_chars >= 100 and avg_conf >= 0.78:
        return False
    if total_chars >= 150 and avg_conf >= 0.72:
        return False

    gray = (
        img_rgb[..., 0].astype(np.float32) * 0.2989
        + img_rgb[..., 1].astype(np.float32) * 0.5870
        + img_rgb[..., 2].astype(np.float32) * 0.1140
    )
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean() if gray.shape[0] > 1 else 0.0
    edge_level = float(gx + gy)

    heights = _word_heights(paddle_words)
    height_cv = _coefficient_of_variation(heights) if len(heights) >= 5 else 0.0
    if len(heights) >= 8 and height_cv > 0.35 and avg_conf < 0.70:
        return True

    conf_cv = _coefficient_of_variation(confs) if len(confs) >= 5 else 0.0
    if len(confs) >= 10 and conf_cv > 0.45 and avg_conf < 0.68:
        return True

    if total_chars >= 30 and avg_conf < 0.72 and edge_level >= 22.0:
        return True
    if total_chars >= 15 and avg_conf < 0.60 and edge_level >= 25.0:
        return True

    return False


def select_best_paddle_result(
    candidates: List[Tuple[List[Dict], str, np.ndarray]]
) -> Tuple[List[Dict], str, np.ndarray]:
    """
    Given multiple (words, preset_name, processed_img) candidates from different
    preprocessing passes, return the one with the highest score_words().
    Tie-break: prefer higher avg_conf, then more words.
    """
    if not candidates:
        return [], "none", np.zeros((1, 1, 3), dtype=np.uint8)

    def sort_key(c: Tuple) -> Tuple[float, float, int]:
        words = c[0]
        return (score_words(words), _avg_conf(words), len(words))

    return max(candidates, key=sort_key)
