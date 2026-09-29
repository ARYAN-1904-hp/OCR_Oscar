"""PPStructure singleton wrapper (PaddleOCR 2.8.x)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import threading
import traceback

import numpy as np

from ocr.paddle_compat import ensure_paddle_cpu_safe


def _quad(bbox: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _view_offset(view: np.ndarray) -> Optional[tuple]:
    """(x, y) of a numpy slice view inside its base image, or None."""
    base = getattr(view, "base", None)
    if not isinstance(base, np.ndarray) or base.ndim != 3 or view.ndim != 3:
        return None
    off = view.__array_interface__["data"][0] - base.__array_interface__["data"][0]
    if off < 0:
        return None
    y, rem = divmod(off, base.strides[0])
    return rem // base.strides[1], y


@contextmanager
def _reuse_words(pipeline: Any, words: List[Dict]):
    """Temporarily point PPStructure's OCR hooks at pre-recognized words."""
    words = [w for w in words if (w.get("text") or "").strip() and w.get("bbox")]

    class _TextSystem:
        def __call__(self, img, *args, **kwargs):
            boxes = [_quad(w["bbox"]) for w in words]
            rec = [(w["text"], float(w.get("conf", 0.0))) for w in words]
            return boxes, rec, {"det": 0.0, "rec": 0.0, "all": 0.0}

    table_system = getattr(pipeline, "table_system", None)
    orig_text_system = getattr(pipeline, "text_system", None)
    orig_table_ocr = getattr(table_system, "_ocr", None) if table_system is not None else None
    crop_offset: List[Optional[tuple]] = [None]

    class _TableSystem:
        """Records where each table crop sits on the page, then delegates."""

        def __getattr__(self, name):
            return getattr(table_system, name)

        def __call__(self, roi_img, *args, **kwargs):
            # StructureSystem passes the crop as a slice of its page copy;
            # TableSystem deep-copies it before _ocr, so read the offset here.
            crop_offset[0] = _view_offset(roi_img)
            try:
                return table_system(roi_img, *args, **kwargs)
            finally:
                crop_offset[0] = None

    def _table_ocr(roi_img):
        offset = crop_offset[0]
        if offset is None:
            return orig_table_ocr(roi_img)
        ox, oy = offset
        h, w = roi_img.shape[:2]
        boxes, rec = [], []
        for wd in words:
            x1, y1, x2, y2 = wd["bbox"][:4]
            cx, cy = (x1 + x2) / 2 - ox, (y1 + y2) / 2 - oy
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            boxes.append([max(0, x1 - ox - 1), max(0, y1 - oy - 1),
                          min(w, x2 - ox + 1), min(h, y2 - oy + 1)])
            rec.append((wd["text"], float(wd.get("conf", 0.0))))
        # TableMatch expects reading order (top-to-bottom, left-to-right).
        order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
        dt_boxes = np.array([boxes[i] for i in order], dtype=np.float32).reshape(-1, 4)
        return dt_boxes, [rec[i] for i in order], 0.0, 0.0

    try:
        if orig_text_system is not None:
            pipeline.text_system = _TextSystem()
        if orig_table_ocr is not None:
            table_system._ocr = _table_ocr
            pipeline.table_system = _TableSystem()
        yield
    finally:
        if orig_text_system is not None:
            pipeline.text_system = orig_text_system
        if orig_table_ocr is not None:
            pipeline.table_system = table_system
            del table_system._ocr  # drop instance override, restore class method


class PPStructureEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.pipeline = None
        self.available = False
        self.last_error = None
        try:
            ensure_paddle_cpu_safe()
            from paddleocr import PPStructure

            self.pipeline = PPStructure(
                show_log=False,
                layout=True,
                table=True,
                ocr=True,
                lang="en",
                enable_mkldnn=False,
                use_gpu=False,
            )
            self.available = True
            print("[PPStructure] Successfully initialized.")
        except Exception as e:
            self.last_error = str(e)
            self.available = False
            print(f"[PPStructure] Initialization failed: {e}")
        self._initialized = True

    def is_available(self) -> bool:
        return self.available

    def analyse(self, image: Any, words: Optional[List[Dict]] = None) -> Optional[Dict[str, Any]]:
        """Run layout + table structure on `image`.

        `words` are PaddleOCR word boxes already recognized on this same image.
        When given, PPStructure reuses them instead of running its own full
        text detection + recognition pass (for text regions and for table
        cells) - that duplicate pass was over half the per-image runtime.
        """
        if not self.available or image is None:
            return None
        try:
            if words is not None:
                with _reuse_words(self.pipeline, words):
                    result = self.pipeline(image)
            else:
                result = self.pipeline(image)
            if result is None:
                raw = []
            elif isinstance(result, list):
                raw = result
            else:
                raw = list(result)
            return {
                "success": True,
                "engine": "ppstructure",
                "raw": raw,
                "metadata": {"regions": len(raw)},
            }
        except Exception as e:
            self.last_error = str(e)
            print("\n========== PPStructure Runtime Error ==========")
            print(self.last_error)
            traceback.print_exc()
            print("===============================================")
            return None


_engine = None


def get_ppstructure() -> PPStructureEngine:
    global _engine
    if _engine is None:
        _engine = PPStructureEngine()
    return _engine


def reset_ppstructure() -> None:
    global _engine
    with PPStructureEngine._lock:
        PPStructureEngine._instance = None
        _engine = None
