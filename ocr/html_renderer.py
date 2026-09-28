"""
Layout HTML preview keyed to source_preview / page image.

Shows the source image exactly as it looks (native pixels, no stretch).
Layout/table boxes are scaled from OCR coordinates into image coordinates.
"""

from __future__ import annotations

import html
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _esc(text: str) -> str:
    return html.escape(text or "", quote=True)


def _bbox(obj: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    b = obj.get("bbox") or []
    if not b or len(b) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in b[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def _scale_bbox(
    bb: Tuple[int, int, int, int],
    sx: float,
    sy: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bb
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def _ensure_image_preview(source_path: Optional[str], folder: Path) -> str:
    """Copy source image into output folder as source_preview.*"""
    if not source_path:
        # Reuse existing preview if present
        for name in (
            "source_preview.jpg",
            "source_preview.jpeg",
            "source_preview.png",
            "source_preview.webp",
        ):
            if (folder / name).exists():
                return name
        return ""

    src = Path(source_path)
    if not src.exists():
        return ""

    suffix = src.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        bg_name = f"source_preview{suffix if suffix != '.jpeg' else '.jpg'}"
        if suffix == ".jpeg":
            bg_name = "source_preview.jpg"
        try:
            shutil.copy2(src, folder / bg_name)
            return bg_name
        except Exception:
            return ""

    if suffix == ".pdf":
        # Render first page (and later pages) to PNG for exact visual reference
        try:
            import fitz
            import numpy as np
            from PIL import Image
        except Exception:
            return ""
        try:
            doc = fitz.open(str(src))
            page = doc.load_page(0)
            # Match common OCR DPI look without huge files
            zoom = 150 / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            name = "source_preview.png"
            Image.fromarray(img).save(folder / name)
            doc.close()
            return name
        except Exception:
            return ""

    return ""


def _render_pdf_pages(
    source_path: Optional[str],
    folder: Path,
    page_count: int,
    dpi: int = 150,
) -> List[str]:
    names: List[str] = []
    if not source_path:
        return names
    src = Path(source_path)
    if not src.exists() or src.suffix.lower() != ".pdf":
        return names
    try:
        import fitz
        import numpy as np
        from PIL import Image
    except Exception:
        return names

    try:
        doc = fitz.open(str(src))
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        n = min(len(doc), page_count)
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            name = f"page_{i + 1}.png"
            Image.fromarray(img).save(folder / name)
            names.append(name)
        doc.close()
    except Exception:
        return []
    return names


def _outline(
    bb: Tuple[int, int, int, int],
    css_class: str,
    title: str = "",
) -> str:
    x1, y1, x2, y2 = bb
    return (
        f'<div class="{css_class}" title="{_esc(title)}" style="'
        f"left:{x1}px;top:{y1}px;width:{max(1, x2 - x1)}px;"
        f'height:{max(1, y2 - y1)}px;"></div>'
    )


def render_layout_html(
    ocr_result: Dict[str, Any],
    output_dir: str | Path,
    source_path: Optional[str] = None,
) -> Path:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    pages = ocr_result.get("pages") or []
    meta = ocr_result.get("meta") or {}
    dpi = int(meta.get("dpi") or 150)

    preview_name = _ensure_image_preview(source_path, folder)
    pdf_pages = []
    if not preview_name and source_path and Path(source_path).suffix.lower() == ".pdf":
        pdf_pages = _render_pdf_pages(source_path, folder, max(1, len(pages)), dpi=dpi)
        if pdf_pages:
            preview_name = pdf_pages[0]
            # also alias first page as source_preview for consistency
            try:
                shutil.copy2(folder / pdf_pages[0], folder / "source_preview.png")
                preview_name = "source_preview.png"
            except Exception:
                pass

    page_html: List[str] = []

    for page in pages:
        page_index = int(page.get("page_index") or 0)
        ocr_w = float(page.get("width") or 0) or 1.0
        ocr_h = float(page.get("height") or 0) or 1.0

        # Choose image for this page — must look like source_preview
        if page_index == 0 and preview_name:
            img_name = preview_name
        elif page_index < len(pdf_pages):
            img_name = pdf_pages[page_index]
        elif page_index == 0:
            img_name = ""
            for cand in (
                "source_preview.jpg",
                "source_preview.jpeg",
                "source_preview.png",
                "source_preview.webp",
            ):
                if (folder / cand).exists():
                    img_name = cand
                    break
        else:
            img_name = f"page_{page_index + 1}.png"
            if not (folder / img_name).exists():
                img_name = ""

        img_w, img_h = int(ocr_w), int(ocr_h)
        if img_name:
            size = _image_size(folder / img_name)
            if size:
                img_w, img_h = size

        # Map OCR boxes → image pixel space (fixes 1000x1149 vs 667x766 mismatch)
        sx = img_w / ocr_w
        sy = img_h / ocr_h

        outlines: List[str] = []
        for i, table in enumerate(page.get("tables") or []):
            if not isinstance(table, dict):
                continue
            bb = _bbox(table)
            if not bb:
                continue
            outlines.append(
                _outline(
                    _scale_bbox(bb, sx, sy),
                    "box-table",
                    f"TABLE {i} ({table.get('source') or 'table'})",
                )
            )

        for i, block in enumerate(page.get("blocks") or []):
            bb = _bbox(block)
            if not bb:
                continue
            btype = (block.get("block_type") or "block").lower()
            if btype == "noise":
                continue
            outlines.append(
                _outline(
                    _scale_bbox(bb, sx, sy),
                    f"box-block type-{btype}",
                    f"BLOCK {i}: {btype}",
                )
            )

        overlay = "\n".join(outlines)

        if img_name:
            # Native image size — identical look to opening source_preview
            body = (
                f'<img class="page-img" src="{_esc(img_name)}" '
                f'width="{img_w}" height="{img_h}" alt="source preview page {page_index + 1}" />'
                f'<div class="overlay" style="width:{img_w}px;height:{img_h}px;">{overlay}</div>'
            )
        else:
            body = f'<pre class="plain">{_esc(page.get("text") or "")}</pre>'

        page_html.append(
            f'<div class="page-wrap" style="width:{img_w}px;">'
            f'<section class="page" style="width:{img_w}px;height:{img_h}px;">'
            f"{body}</section>"
            f"</div>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>OCR Layout Preview</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 20px;
      background: #dfe6eb;
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 15px;
      color: #24343d;
    }}
    .hint {{
      margin: 0 0 14px;
      font-size: 12px;
      color: #5a6d78;
    }}
    .page-wrap {{
      margin: 0 auto 24px;
      background: #fff;
      box-shadow: 0 2px 10px rgba(0,0,0,.15);
      line-height: 0;
    }}
    .page {{
      position: relative;
      margin: 0 auto;
      background: #fff;
    }}
    .page-img {{
      display: block;
      width: 100%;
      height: auto;
      /* never stretch/squash — page box matches image pixel size */
    }}
    .overlay {{
      position: absolute;
      left: 0;
      top: 0;
      pointer-events: none;
    }}
    .box-table, .box-block {{
      position: absolute;
      pointer-events: none;
    }}
    .box-table {{
      border: 2px solid rgba(20, 140, 90, 0.95);
      background: rgba(20, 140, 90, 0.08);
    }}
    .box-block {{
      border: 1px solid rgba(25, 105, 160, 0.75);
      background: rgba(25, 105, 160, 0.04);
    }}
    .box-block.type-heading,
    .box-block.type-header {{
      border-color: rgba(110, 70, 170, 0.85);
      background: rgba(110, 70, 170, 0.05);
    }}
    .plain {{
      margin: 16px;
      line-height: 1.4;
      white-space: pre-wrap;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <h1>OCR Layout Preview</h1>
  <p class="hint">Reference image shown exactly as <code>source_preview</code> (native size, no stretch). Boxes are scaled to that image.</p>
  {"".join(page_html) if page_html else "<p>No pages in OCR result.</p>"}
</body>
</html>
"""
    # Fix accidental empty concat in CSS from draft
    doc = doc.replace("width: {\"\" and \"\"}100%;", "width: 100%;")

    out = folder / "layout.html"
    out.write_text(doc, encoding="utf-8")
    return out
