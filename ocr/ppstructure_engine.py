"""PPStructure singleton wrapper (PaddleOCR 2.8.x)."""

from __future__ import annotations

from typing import Any, Dict, Optional
import threading
import traceback

from ocr.paddle_compat import ensure_paddle_cpu_safe


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

    def analyse(self, image: Any) -> Optional[Dict[str, Any]]:
        if not self.available or image is None:
            return None
        try:
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
