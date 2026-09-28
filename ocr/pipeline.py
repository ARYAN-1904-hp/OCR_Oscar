from __future__ import annotations

"""
Fast pipeline for raw JSON output (pages/lines/words). Parser handles layout.

- PaddleOCR for word boxes
- preset=auto uses multi-pass preprocessing. No tables/header_footer/layout/fields/analysis.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from ocr.paddle_compat import ensure_paddle_cpu_safe  # noqa: E402

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

from ocr.pdf_utils import render_pdf_to_images  # noqa: E402
from ocr.preprocess import preprocess  # noqa: E402
from ocr.postprocess import normalize_linebreaks  # noqa: E402
from ocr.schemas import Word, Line, PageResult, to_dict  # noqa: E402
from ocr.engine_router import score_words, select_best_paddle_result  # noqa: E402
from ocr.hybrid_layout import HybridLayoutEngine, route_pdf_document  # noqa: E402
from ocr.form_postprocess import enhance_page_forms  # noqa: E402
from ocr.recognition_refine import (  # noqa: E402
    drop_icon_glyphs,
    recover_empty_field_values,
    refine_word_recognition,
)
from ocr.office_utils import (  # noqa: E402
    docx_load,
    pptx_extract_slides,
    xlsx_load,
    convert_office_to_pdf,
)

def poly_to_xyxy(poly: Any) -> List[int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def merge_line_bbox(words: List[dict]) -> List[int]:
    return [
        int(min(w["bbox"][0] for w in words)),
        int(min(w["bbox"][1] for w in words)),
        int(max(w["bbox"][2] for w in words)),
        int(max(w["bbox"][3] for w in words)),
    ]

def compute_line_conf(words: List[dict]) -> float:
    return float(sum(float(w.get("conf", 0.0)) for w in words) / max(1, len(words))) if words else 0.0

def line_text(words: List[dict]) -> str:
    ws = sorted(words, key=lambda w: (w["bbox"][0], w["bbox"][1]))
    return " ".join((w.get("text") or "").strip() for w in ws).strip()

def group_into_lines(words: List[dict]) -> List[List[dict]]:
    if not words:
        return []

    def y_center(w: dict) -> float:
        return (w["bbox"][1] + w["bbox"][3]) / 2.0

    ws = sorted(words, key=lambda w: (y_center(w), w["bbox"][0]))
    heights = sorted(max(1, w["bbox"][3] - w["bbox"][1]) for w in ws)
    med_h = heights[len(heights) // 2] if heights else 12
    thr = max(8.0, med_h * 0.65)

    lines: List[List[dict]] = []
    current: List[dict] = []
    current_y: Optional[float] = None

    for w in ws:
        yc = y_center(w)
        if current_y is None:
            current = [w]
            current_y = yc
            continue
        if abs(yc - current_y) <= thr:
            current.append(w)
            current_y = (current_y * (len(current) - 1) + yc) / len(current)
        else:
            lines.append(sorted(current, key=lambda x: x["bbox"][0]))
            current = [w]
            current_y = yc

    if current:
        lines.append(sorted(current, key=lambda x: x["bbox"][0]))
    return lines

_OCR_INSTANCE: Optional[Any] = None
_LAYOUT_ENGINE: Optional[HybridLayoutEngine] = None


def _get_layout_engine() -> HybridLayoutEngine:
    global _LAYOUT_ENGINE
    if _LAYOUT_ENGINE is None:
        _LAYOUT_ENGINE = HybridLayoutEngine()
    return _LAYOUT_ENGINE


def warmup_models(load_ppstructure: bool = True) -> None:
    """Eagerly load the PaddleOCR (and optionally PPStructure) singletons.

    Call once at service startup so the first incoming request doesn't pay
    model load time; both `get_paddle()` and `get_ppstructure()` are
    otherwise lazy and would initialize on first use.
    """
    get_paddle()
    if load_ppstructure:
        from ocr.ppstructure_engine import get_ppstructure

        get_ppstructure()


def get_paddle() -> Any:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE

    ensure_paddle_cpu_safe()
    from paddleocr import PaddleOCR  # lazy import

    common_kwargs = dict(
        use_angle_cls=False,  # angle cls nearly doubles runtime; enable via OCR_ANGLE_CLS=1 if needed
        lang="en",
        show_log=False,
        det_db_score_mode="fast",
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        # PaddleOCR's DB detector internally downscales the input so its
        # longest side fits this limit (default 960) before proposing text
        # boxes. preprocess.py upscales pages to >=1500px on the long side
        # for recognition quality, but without raising this too, the
        # detector silently shrinks that back down to 960 - a small,
        # isolated single-character value (e.g. a lone "Gender: F") falls
        # below the box-score threshold at that scale and never gets a box
        # proposed at all, so it's dropped before recognition/postprocess
        # ever see it.
        # 1280 is a deliberate middle ground, not the ceiling: pushing this
        # all the way up to MAX_OCR_DIM (1600+) does recover isolated single
        # chars at higher confidence, but also measurably tightens box
        # proposals for ordinary short words - e.g. it clipped "Term" down
        # to "Tem" in testing on a real claims form. 1280 was the highest
        # value that still detected the isolated single character without
        # reproducing that word-clipping regression on the same page.
        det_limit_side_len=1280,
        rec_batch_num=16,
        use_gpu=False,
        enable_mkldnn=False,
    )
    if os.environ.get("OCR_ANGLE_CLS", "0").strip().lower() in ("1", "true", "yes", "on"):
        common_kwargs["use_angle_cls"] = True

    for extra in [
        {"use_mkldnn": False},
        {},
    ]:
        try:
            _OCR_INSTANCE = PaddleOCR(**common_kwargs, **extra)
            return _OCR_INSTANCE
        except TypeError:
            continue

    _OCR_INSTANCE = PaddleOCR(
        use_angle_cls=bool(common_kwargs.get("use_angle_cls")),
        lang="en",
        show_log=False,
        use_gpu=False,
        enable_mkldnn=False,
    )
    return _OCR_INSTANCE

# Max dimension for OCR (larger images are downscaled for speed)
MAX_OCR_DIM = int(os.environ.get("OCR_MAX_DIM", "1600"))


def _downscale_if_large(img_rgb: np.ndarray, max_dim: int = MAX_OCR_DIM) -> np.ndarray:
    """Downscale image if either side exceeds max_dim to speed up OCR."""
    h, w = img_rgb.shape[:2]
    if max(h, w) <= max_dim:
        return img_rgb
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w < 1 or new_h < 1:
        return img_rgb
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    out = np.array(Image.fromarray(img_rgb).resize((new_w, new_h), resample))
    return out


def _read_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _read_image_pages(path: str, suffix: str, max_pages: int) -> List[np.ndarray]:
    """Load image(s): single page for most formats, multiple frames for TIFF."""
    suffix = (suffix or "").lower()
    if suffix in (".tiff", ".tif"):
        im = Image.open(path)
        frames: List[np.ndarray] = []
        n_frames = getattr(im, "n_frames", 1)
        n_frames = min(n_frames, max_pages)
        for i in range(n_frames):
            im.seek(i)
            frames.append(np.array(im.convert("RGB")))
        return frames if frames else [_read_image(path)]
    return [_read_image(path)]

def _paddle_words_on_image(img_rgb: np.ndarray) -> List[dict]:
    ocr = get_paddle()
    use_cls = bool(getattr(ocr, "use_angle_cls", False))
    result = ocr.ocr(img_rgb, cls=use_cls)

    words: List[dict] = []
    if result is None:
        return words

    items = result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        if len(result[0]) == 0:
            return words
        first = result[0][0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            items = result[0]

    for item in items:
        if not item:
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            poly = item[0]
            tc = item[1]
            if isinstance(tc, (list, tuple)) and len(tc) >= 2:
                text = str(tc[0])
                conf = float(tc[1])
            else:
                continue
            try:
                bbox = poly_to_xyxy(poly)
            except Exception:
                continue
            words.append({"text": text, "conf": conf, "bbox": bbox})
    return words


def _refine_words(img_rgb: np.ndarray, words: List[dict]) -> List[dict]:
    """Recover punctuation/spacing clipped by tight det boxes and values the detector missed.

    Disable with OCR_REFINE=0.
    """
    if os.environ.get("OCR_REFINE", "1").strip().lower() in ("0", "false", "no", "off"):
        return words
    ocr = get_paddle()
    # Icons are judged on the detector's own read: padding can push a "0x"
    # checkbox/close-icon read to a confident-looking "0 x".
    words = drop_icon_glyphs(words)
    words = refine_word_recognition(ocr, img_rgb, words)
    return recover_empty_field_values(ocr, img_rgb, words)

def _ocr_best_of_presets(img_rgb: np.ndarray, preset: str) -> Tuple[List[dict], str, np.ndarray]:
    """
    Run PaddleOCR on a preprocessed image.

    Default is a single preprocess+OCR pass (fast). Multi-pass best-of is
    opt-in via OCR_MULTI_PASS=1 because it runs OCR 2–3x per image.
    """
    img_rgb = _downscale_if_large(img_rgb)
    multi_pass = os.environ.get("OCR_MULTI_PASS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if preset == "auto" and multi_pass:
        from ocr.preprocess import get_all_presets_for_auto

        candidates = []
        for preset_name, proc_img in get_all_presets_for_auto(img_rgb):
            words = _paddle_words_on_image(proc_img)
            # Early exit when a pass is already high quality
            if score_words(words) >= 4.0 and len(words) >= 20:
                return _refine_words(proc_img, words), preset_name, proc_img
            candidates.append((words, preset_name, proc_img))
        words, preset_name, proc_img = select_best_paddle_result(candidates)
        return _refine_words(proc_img, words), preset_name, proc_img

    processed, used = preprocess(img_rgb, preset=preset)
    words = _paddle_words_on_image(processed)
    return _refine_words(processed, words), used, processed


def _ocr_one_image(
    img_rgb: np.ndarray,
    page_idx: int,
    preset: str,
    engine: str,
    engine_used_first: Optional[str],
    return_debug: bool,
    use_ppstructure: bool = False,
) -> Tuple[PageResult, str, str, Optional[dict]]:
    """Run OCR on one image; return (PageResult, preset_used, engine_used, debug)."""

    paddle_words, used_preset, used_img = _ocr_best_of_presets(img_rgb, preset=preset)
    _ = engine  # retained for API compatibility
    engine_used = "paddleocr"
    if page_idx == 0:
        engine_used_first = engine_used

    h, w = used_img.shape[:2]

    # -- Hybrid layout (heuristic + optional PPStructure for tables) ------
    blocks, columns, tables = _get_layout_engine().analyse(
        words=paddle_words,
        page_width=w,
        page_height=h,
        image=used_img,
        use_ppstructure=use_ppstructure,
    )

    # -- Build flat Line list (backward compat) from layout blocks --------
    all_lines: List[Line] = []
    page_text_parts: List[str] = []

    for block in blocks:
        if block.block_type == "noise":
            continue
        for ln in block.lines:
            if ln.text.strip():
                all_lines.append(ln)
                page_text_parts.append(ln.text)

    page_text = "\n".join(page_text_parts).strip()

    pr = PageResult(
        page_index=page_idx,
        width=w,
        height=h,
        text=page_text,
        lines=all_lines,
        columns=columns,
        blocks=blocks,
        tables=tables,
    )
    # Checkbox ink detection + two-column form / table cleanup
    pr = enhance_page_forms(pr, image_rgb=used_img)

    debug_entry: Optional[dict] = None
    if return_debug:
        debug_entry = {
            "page_index": page_idx,
            "preset_used": used_preset,
            "engine_used": engine_used,
            "words_count": len(paddle_words),
            "lines_count": len(all_lines),
            "blocks_count": len(blocks),
            "columns_count": len(columns),
            "tables_count": len(pr.tables or []),
            "checkboxes_count": len(pr.checkboxes or []),
            "block_types": {
                bt: sum(1 for b in blocks if b.block_type == bt)
                for bt in set(b.block_type for b in blocks)
            },
        }

    return pr, used_preset, engine_used_first or engine_used, debug_entry


def run_ocr(
    input_path: str,
    preset: str = "auto",
    dpi: int = 200,
    max_pages: int = 5,
    return_debug: bool = True,
    engine: str = "auto",
    return_layout: bool = True,
) -> Dict[str, Any]:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    suffix = p.suffix.lower()
    input_path = str(p)
    converted_pdf_path: Optional[str] = None

    # DOC/PPT: convert to PDF via LibreOffice, then process as PDF
    if suffix in (".doc", ".ppt"):
        pdf_path = convert_office_to_pdf(input_path)
        if not pdf_path:
            raise FileNotFoundError(
                "DOC/PPT conversion failed. Install LibreOffice and ensure 'soffice' is in PATH, "
                "or upload PDF/DOCX/PPTX instead."
            )
        converted_pdf_path = pdf_path
        input_path = pdf_path
        p = Path(input_path)
        suffix = ".pdf"

    meta: Dict[str, Any] = {
        "engine": "paddleocr",
        "lang": "en",
        "preset_requested": preset,
        "dpi": dpi,
        "max_pages": max_pages,
        "input_type": (
            "pdf" if suffix == ".pdf"
            else "docx" if suffix == ".docx"
            else "pptx" if suffix == ".pptx"
            else "xlsx" if suffix == ".xlsx"
            else "image"
        ),
        "mkldnn_disabled": True,
        "engine_requested": engine,
        "return_layout": bool(return_layout),
        "layout_engine": "HybridLayoutEngine",
        # Do not instantiate HybridLayoutEngine here (avoids slow PPStructure init
        # on the digital-PDF fast path).
        "ppstructure": os.environ.get("OCR_USE_PPSTRUCTURE", "0").strip().lower()
        in ("1", "true", "yes", "on"),
    }

    try:
        if suffix == ".pdf":
            # Digital-first routing: inspect each page's native text layer
            # (PyMuPDF) before rasterizing anything. Digital pages are
            # extracted directly via PyMuPDF + pdfplumber, bypassing
            # PaddleOCR entirely; only scanned/image-based pages are
            # rasterized and OCR'd. Force all pages through visual OCR
            # (skip routing) with OCR_FORCE_VISUAL=1.
            force_visual = os.environ.get("OCR_FORCE_VISUAL", "0").strip().lower() in (
                "1", "true", "yes", "on",
            )
            if not force_visual:
                pp_env = os.environ.get("OCR_USE_PPSTRUCTURE", "auto").strip().lower()
                if pp_env in ("0", "false", "no", "off"):
                    use_pp_for_scanned = False
                elif pp_env in ("1", "true", "yes", "on"):
                    use_pp_for_scanned = True
                else:
                    use_pp_for_scanned = False  # PDFs default to the fast path

                def _route_ocr_page(img_rgb: np.ndarray, page_idx: int):
                    pr, used_preset, _engine_used, de = _ocr_one_image(
                        img_rgb, page_idx, preset, engine, None, return_debug,
                        use_ppstructure=use_pp_for_scanned,
                    )
                    meta["preset_used"] = used_preset
                    return pr, (de or {})

                pages, route_debug = route_pdf_document(
                    str(p), ocr_page_fn=_route_ocr_page, max_pages=max_pages, dpi=dpi,
                )
                n_digital = sum(1 for d in route_debug if d.get("route") == "digital")
                n_scanned = len(route_debug) - n_digital

                for pg, dbg in zip(pages, route_debug):
                    if dbg.get("route") == "digital":
                        enhance_page_forms(pg, image_rgb=None)
                    # Scanned pages are already enhanced (with their raster
                    # image) inside _ocr_one_image - don't run it twice.

                meta["digital_first_routing"] = True
                meta["digital_pdf"] = n_digital > 0
                meta["layout_visual_ocr"] = n_scanned > 0
                meta["layout_engine"] = (
                    "DigitalPdfLayout" if n_scanned == 0 else "HybridLayoutEngine+DigitalPdfLayout"
                )
                meta["pages_digital"] = n_digital
                meta["pages_scanned"] = n_scanned
                meta["engine_used"] = (
                    "digital_pdf" if n_scanned == 0
                    else "paddleocr" if n_digital == 0
                    else "digital_pdf+paddleocr"
                )
                meta["ppstructure"] = bool(use_pp_for_scanned and n_scanned > 0)

                out = to_dict(pages, meta)
                out["text"] = normalize_linebreaks(out.get("text") or "")
                for pg in out.get("pages", []):
                    pg["text"] = normalize_linebreaks(pg.get("text") or "")
                if return_debug:
                    out["debug"] = route_debug
                return out

            meta["digital_first_routing"] = False
            meta["digital_pdf"] = False
            meta["layout_visual_ocr"] = True
            imgs = render_pdf_to_images(str(p), dpi=dpi, max_pages=max_pages)
        elif suffix == ".docx":
            full_text, images = docx_load(input_path, max_pages=max_pages)
            page_results = [PageResult(page_index=0, width=0, height=0, text=full_text or "", lines=[])]
            debug_info = []
            engine_used_first: Optional[str] = None
            last_preset = "auto"
            for i, img in enumerate(images):
                pr, used_preset, engine_used_first, de = _ocr_one_image(
                    img, len(page_results), preset, engine, engine_used_first, return_debug
                )
                page_results.append(pr)
                last_preset = used_preset
                if return_debug and de:
                    debug_info.append(de)
            meta["preset_used"] = last_preset
            meta["engine_used"] = engine_used_first or ""
            out = to_dict(page_results, meta)
            out["text"] = normalize_linebreaks(out.get("text") or "")
            for pg in out.get("pages", []):
                pg["text"] = normalize_linebreaks(pg.get("text") or "")
            if return_debug:
                out["debug"] = debug_info
            return out
        elif suffix == ".pptx":
            slides = pptx_extract_slides(input_path, max_slides=max_pages)
            page_results = []
            debug_info = []
            engine_used_first = None
            last_preset = "auto"
            for slide_idx, (slide_text, slide_images) in enumerate(slides):
                if slide_images:
                    pr, used_preset, engine_used_first, de = _ocr_one_image(
                        slide_images[0], slide_idx, preset, engine, engine_used_first, return_debug
                    )
                    combined = (slide_text + "\n" + pr.text).strip() if slide_text else pr.text
                    page_results.append(
                        PageResult(
                            page_index=pr.page_index,
                            width=pr.width,
                            height=pr.height,
                            text=combined,
                            lines=pr.lines,
                            columns=pr.columns,
                            blocks=pr.blocks,
                            tables=pr.tables,
                        )
                    )
                    last_preset = used_preset
                    if return_debug and de:
                        debug_info.append(de)
                else:
                    page_results.append(PageResult(slide_idx, 0, 0, slide_text or "", []))
            meta["preset_used"] = last_preset
            meta["engine_used"] = engine_used_first or ""
            out = to_dict(page_results, meta)
            out["text"] = normalize_linebreaks(out.get("text") or "")
            for pg in out.get("pages", []):
                pg["text"] = normalize_linebreaks(pg.get("text") or "")
            if return_debug:
                out["debug"] = debug_info
            return out
        elif suffix == ".xlsx":
            sheets, images = xlsx_load(input_path, max_pages=max_pages)
            page_results = [
                PageResult(page_index=i, width=0, height=0, text=sheet_text or "", lines=[])
                for i, (_sheet_name, sheet_text) in enumerate(sheets)
            ]
            debug_info = []
            engine_used_first = None
            last_preset = "auto"
            for img in images:
                pr, used_preset, engine_used_first, de = _ocr_one_image(
                    img, len(page_results), preset, engine, engine_used_first, return_debug
                )
                page_results.append(pr)
                last_preset = used_preset
                if return_debug and de:
                    debug_info.append(de)
            meta["preset_used"] = last_preset
            meta["engine_used"] = engine_used_first or ""
            meta["sheet_names"] = [name for name, _text in sheets]
            out = to_dict(page_results, meta)
            out["text"] = normalize_linebreaks(out.get("text") or "")
            for pg in out.get("pages", []):
                pg["text"] = normalize_linebreaks(pg.get("text") or "")
            if return_debug:
                out["debug"] = debug_info
            return out
        else:
            imgs = _read_image_pages(str(p), suffix, max_pages)

        page_results = []
        debug_info = []
        engine_used_first = None

        # Images: enable PPStructure for real table structure (single-page forms).
        # PDFs stay on digital/fast path unless OCR_FORCE_VISUAL=1.
        # Override with OCR_USE_PPSTRUCTURE=0/1.
        pp_env = os.environ.get("OCR_USE_PPSTRUCTURE", "auto").strip().lower()
        if pp_env in ("0", "false", "no", "off"):
            use_pp = False
        elif pp_env in ("1", "true", "yes", "on"):
            use_pp = True
        else:
            use_pp = meta.get("input_type") == "image"

        meta["ppstructure"] = bool(use_pp)

        for page_idx, img_rgb in enumerate(imgs):
            pr, used_preset, engine_used_first, de = _ocr_one_image(
                img_rgb,
                page_idx,
                preset,
                engine,
                engine_used_first,
                return_debug,
                use_ppstructure=use_pp,
            )
            page_results.append(pr)
            meta["preset_used"] = used_preset
            if return_debug and de:
                debug_info.append(de)
        meta["engine_used"] = engine_used_first or ""
        if use_pp:
            meta["engine_used"] = f"{meta['engine_used']}+ppstructure"

        out = to_dict(page_results, meta)

        out["text"] = normalize_linebreaks(out.get("text") or "")
        for pg in out.get("pages", []):
            pg["text"] = normalize_linebreaks(pg.get("text") or "")

        if return_debug:
            out["debug"] = debug_info

        return out
    finally:
        if converted_pdf_path:
            try:
                Path(converted_pdf_path).unlink(missing_ok=True)
            except Exception:
                pass
