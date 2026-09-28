from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_CLAHE_STRONG = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
_SHARPEN_KERNEL = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], dtype=np.float32)
_SHARPEN_MILD = np.array([[-0.5, -0.5, -0.5],
                           [-0.5, 5.0, -0.5],
                           [-0.5, -0.5, -0.5]], dtype=np.float32) / 1.0


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale back to RGB (3-channel) for PaddleOCR."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _clahe(gray: np.ndarray, strong: bool = False) -> np.ndarray:
    engine = _CLAHE_STRONG if strong else _CLAHE
    return engine.apply(gray)


def _adaptive_thresh(gray: np.ndarray, block_size: int = 31, C: int = 10) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )


def _denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(gray, ksize)


def _bilateral_denoise(img_rgb: np.ndarray) -> np.ndarray:
    """Bilateral filter preserves edges while smoothing noise - good for PaddleOCR."""
    return cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)


def _sharpen(gray: np.ndarray, mild: bool = False) -> np.ndarray:
    kernel = _SHARPEN_MILD if mild else _SHARPEN_KERNEL
    return cv2.filter2D(gray, -1, kernel)


def _estimate_skew_angle_small(bin_img: np.ndarray) -> float:
    coords = np.column_stack(np.where(bin_img < 128))
    if coords.shape[0] < 1500:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = float(rect[-1])
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 12:
        return 0.0
    return angle


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


_MIN_LONG_SIDE = 1500  # Lanczos target: longest side (px)


def _unsharp_light(img_rgb: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    """Light unsharp mask — sharpens text edges without amplifying noise much."""
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(img_rgb, 1.0 + amount, blurred, -amount, 0)


def _upscale_rgb(img_rgb: np.ndarray, min_long_side: int = _MIN_LONG_SIDE) -> np.ndarray:
    """
    Lanczos upscale so the longest side reaches min_long_side (if smaller),
    then apply a light unsharp mask. No-op when already large enough.
    """
    h, w = img_rgb.shape[:2]
    long_side = max(h, w)
    if long_side >= min_long_side or long_side == 0:
        return img_rgb
    scale = min_long_side / float(long_side)
    out = cv2.resize(
        img_rgb,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return _unsharp_light(out)


def preprocess(img_rgb: np.ndarray, preset: str = "auto") -> Tuple[np.ndarray, str]:
    """
    Returns preprocessed RGB image and the preset name used.
    PaddleOCR works best with RGB input - always return RGB.

    Presets:
      auto         -> detect and dispatch
      clean_doc    -> deskew + mild denoise + CLAHE (clean printed docs)
      printed_hq   -> bilateral denoise + mild sharpen, NO binarization (PaddleOCR-native)
      photo        -> upscale + denoise + CLAHE + deskew
      low_light    -> strong CLAHE + upscale + deskew
      scan_enhance -> strong CLAHE + adaptive threshold + morphological cleanup

    Resolution: all presets finish with Lanczos upscale-to-min-size + light unsharp
    when the longest side is below the target (default 1500px).
    """
    preset = (preset or "auto").lower().strip()
    valid = {"auto", "clean_doc", "printed_hq", "photo", "low_light", "scan_enhance"}
    if preset not in valid:
        preset = "auto"

    gray = _to_gray(img_rgb)
    used = preset

    if preset == "auto":
        mean = float(gray.mean())
        std = float(gray.std())
        if mean < 80:
            used = "low_light"
        elif mean < 130 and std < 45:
            used = "scan_enhance"
        elif std > 75:
            used = "photo"
        else:
            used = "printed_hq"

    # --- printed_hq: best default for PaddleOCR on clean printed text ---
    if used == "printed_hq":
        out = _bilateral_denoise(img_rgb)
        g = _to_gray(out)
        g = _sharpen(g, mild=True)
        b = _adaptive_thresh(_clahe(_denoise(g)))
        angle = _estimate_skew_angle_small(b)
        out = _rotate(out, angle)
        return _upscale_rgb(out), used

    # --- clean_doc: deskew + denoise + CLAHE (good for forms) ---
    if used == "clean_doc":
        g = _denoise(gray)
        g = _clahe(g)
        b = _adaptive_thresh(g)
        angle = _estimate_skew_angle_small(b)
        out = _rotate(img_rgb, angle)
        return _upscale_rgb(out), used

    # --- photo: upscale + bilateral + CLAHE + deskew ---
    if used == "photo":
        out = _bilateral_denoise(img_rgb)
        out = _upscale_rgb(out)
        g2 = _to_gray(out)
        b = _adaptive_thresh(_clahe(_denoise(g2)))
        angle = _estimate_skew_angle_small(b)
        out = _rotate(out, angle)
        return out, used

    # --- scan_enhance: aggressive for poor scans ---
    if used == "scan_enhance":
        g = _denoise(gray, ksize=5)
        g = _clahe(g, strong=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
        b = _adaptive_thresh(g, block_size=25, C=8)
        angle = _estimate_skew_angle_small(b)
        out_gray = _rotate(g, angle)
        out = _to_rgb(out_gray)
        return _upscale_rgb(out), used

    # --- low_light ---
    g = _clahe(_denoise(gray), strong=True)
    b = _adaptive_thresh(g)
    angle = _estimate_skew_angle_small(b)
    out = _rotate(img_rgb, angle)
    return _upscale_rgb(out), used


def get_all_presets_for_auto(img_rgb: np.ndarray):
    """
    For multi-pass OCR: return preprocessed images under each candidate preset.
    Used in pipeline.py to run PaddleOCR on multiple preprocessed versions
    and pick the best result.
    Returns list of (preset_name, processed_img_rgb) tuples.
    """
    gray = _to_gray(img_rgb)
    mean = float(gray.mean())
    std = float(gray.std())

    candidates = ["printed_hq", "clean_doc"]

    if mean < 80:
        candidates = ["low_light", "scan_enhance", "printed_hq"]
    elif mean < 130 and std < 45:
        candidates = ["scan_enhance", "clean_doc", "printed_hq"]
    elif std > 75:
        candidates = ["photo", "printed_hq"]

    results = []
    for p in candidates:
        processed, used = preprocess(img_rgb, preset=p)
        results.append((used, processed))
    return results
