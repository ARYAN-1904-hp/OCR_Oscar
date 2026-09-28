"""
Shared helpers for tests / workflow:
  • Discover documents in tests/data (PDF + images)
  • Reset runtime storage for clean batch runs
"""

import shutil
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parent / "data"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_GLOBS = (
    "*.pdf",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.tif",
    "*.tiff",
)


def get_sample_documents() -> List[Path]:
    """Return all supported documents in tests/data, sorted by name."""

    docs: List[Path] = []
    for pattern in DOCUMENT_GLOBS:
        docs.extend(DATA_DIR.glob(pattern))

    unique = sorted({p.resolve() for p in docs}, key=lambda p: p.name.lower())

    if not unique:
        raise FileNotFoundError(
            f"No PDF/image found in {DATA_DIR}. "
            "Place one or more .pdf / .jpg / .png files in tests/data and re-run."
        )

    return unique


def get_sample_pdfs() -> List[Path]:
    """Backward-compatible alias: all sample documents (PDF + images). """
    return get_sample_documents()


def get_sample_pdf() -> Path:
    """Return the first sample document in tests/data."""
    return get_sample_documents()[0]


def reset_workspace() -> None:
    """Clear runtime artifacts so a workflow starts clean."""

    from handler_checker.config import (
        ORIGINALS_DIR,
        OUTPUTS_DIR,
        METADATA_FILE,
        QUEUE_FILE,
    )

    for folder in (ORIGINALS_DIR, OUTPUTS_DIR):
        if folder.exists():
            try:
                shutil.rmtree(folder)
            except OSError:
                shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)

    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text("{}", encoding="utf-8")

    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text("[]", encoding="utf-8")

    for junk in (
        PROJECT_ROOT / "ocr_output",
        PROJECT_ROOT / "tests" / "output_test",
    ):
        if junk.exists():
            if junk.is_dir():
                shutil.rmtree(junk)
            else:
                junk.unlink()
