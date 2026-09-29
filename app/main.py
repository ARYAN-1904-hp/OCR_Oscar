from __future__ import annotations

import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ocr.pipeline import run_ocr, warmup_models
from app.parsers.fast_cpu_parser import FastLocalCPUParser
from handler_checker.config import ORIGINALS_DIR, OUTPUTS_DIR, TEXT_COLLECTIONS_DIR
from handler_checker.metadata_manager import MetadataManager
from handler_checker.models import Document
from handler_checker.ocr_manager import OCRManager
from handler_checker.queue_manager import QueueManager


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".doc", ".docx",
    ".ppt", ".pptx",
    ".xlsx",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tiff", ".tif",
    ".gif",
}
ALLOWED_EXTENSIONS_STR = ", ".join(sorted(ALLOWED_EXTENSIONS))
_DOC_ID_RE = re.compile(r"^DOC_\d+$")
_BATCH_ID_RE = re.compile(r"^BATCH_\d+$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
_PARSER_EXTS = {".doc", ".docx", ".ppt", ".pptx", ".xlsx"}

app = FastAPI(
    title="OCR Agent",
    description="PaddleOCR-powered extraction API. Upload a file, get structured text back.",
    version="2.0.0",
)

# Allow frontend (e.g. http://127.0.0.1:5500) to call this API. OCR logic unchanged.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loads once at startup
FAST_PARSER = FastLocalCPUParser()
OCR_MANAGER = OCRManager()
QUEUE_MANAGER = QueueManager()
METADATA_MANAGER = MetadataManager()


@app.on_event("startup")
def _load_ocr_models() -> None:
    """Initialize PaddleOCR/PPStructure singletons once, not per request."""
    warmup_models()


def _check_extension(filename: str) -> None:
    suffix = Path(filename or "upload").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS_STR}",
        )


def _new_document_id() -> str:
    return f"DOC_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


def _document_type_and_pipeline(extension: str) -> tuple[str, str]:
    if extension == ".pdf":
        return "PDF", "PDF_OCR"
    if extension in _IMAGE_EXTS:
        return "IMAGE", "IMAGE_OCR"
    if extension in _PARSER_EXTS:
        return "PARSER", "PARSER"
    return "UNKNOWN", "UNKNOWN"


def _apply_ocr_metadata(document: Document, ocr_result: Dict[str, Any]) -> None:
    """Mirror handler OCR metadata enrichment for frontend uploads."""
    meta = ocr_result.get("meta") or {}
    document.ocr_status = "completed"
    if meta.get("digital_pdf"):
        document.ocr_engine = meta.get("engine_used") or "digital_pdf+layout"
    else:
        base_engine = (
            meta.get("engine_used")
            or meta.get("engine")
            or "paddleocr"
        )
        if meta.get("ppstructure"):
            document.ocr_engine = f"{base_engine}+ppstructure"
        else:
            document.ocr_engine = base_engine
    document.ocr_pages = len(ocr_result.get("pages", []))
    document.ocr_text_length = len(ocr_result.get("text", ""))
    document.processed_pages = document.ocr_pages
    document.original_pages = document.original_pages or document.ocr_pages
    document.validate()


def _persist_frontend_upload(
    filename: str,
    source_path: Path,
    raw_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Persist frontend uploads like the handler path:
      - originals / outputs under handler_checker/storage/
      - queue.json status updates
      - global + local metadata.json
      - layout.html via OCRManager.save_ocr_output
    """
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    document_id = _new_document_id()
    safe_name = Path(filename or "upload").name
    extension = Path(safe_name).suffix.lower()
    doc_type, pipeline = _document_type_and_pipeline(extension)

    document = Document(
        document_id=document_id,
        filename=safe_name,
        extension=extension,
        document_type=doc_type,
        original_path=str(source_path),
        pipeline=pipeline,
        status="UPLOADED",
    )

    QUEUE_MANAGER.enqueue(document.document_id, status="uploaded")
    METADATA_MANAGER.add_document(document)

    storage_path = ORIGINALS_DIR / f"{document_id}_{safe_name}"
    shutil.copy2(source_path, storage_path)
    document.storage_path = str(storage_path)

    output_dir = OUTPUTS_DIR / document_id
    output_dir.mkdir(parents=True, exist_ok=True)
    document.output_path = str(output_dir)

    document.status = "STORED"
    QUEUE_MANAGER.update_status(document.document_id, "stored")
    METADATA_MANAGER.update_document(document)

    # Frontend path: keep simple ongoing → completed (no ready_for_checker)
    document.status = "ONGOING"
    QUEUE_MANAGER.update_status(document.document_id, "ongoing")
    METADATA_MANAGER.update_document(document)

    OCR_MANAGER.save_ocr_output(
        raw_json,
        str(output_dir),
        source_path=str(storage_path),
        source_label=safe_name,
    )
    _apply_ocr_metadata(document, raw_json)

    document.mark_completed()  # status = COMPLETED
    QUEUE_MANAGER.update_status(document.document_id, "completed")
    METADATA_MANAGER.update_document(document)

    storage = {
        "document_id": document_id,
        "storage_path": str(storage_path),
        "output_path": str(output_dir),
        "layout_html": str(output_dir / "layout.html"),
        "layout_html_url": f"/ocr/outputs/{document_id}/layout.html",
        "queue_status": QUEUE_MANAGER.get_status(document_id),
        "metadata_file": str(METADATA_MANAGER.metadata_file),
        "queue_file": str(QUEUE_MANAGER.queue_file),
        "files": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
    }
    return storage


class TextCollection:
    """
    Writes each document's cleaned text to its own <filename>.txt, one folder
    (+ zip) per request.

    Files are written as soon as each document finishes OCR, so the folder
    fills up while the batch runs.
    """

    def __init__(self) -> None:
        self.batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.folder = TEXT_COLLECTIONS_DIR / self.batch_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.files: List[str] = []
        self.errors: List[str] = []

    def _unique_name(self, filename: str) -> str:
        stem = re.sub(r"[^\w.\-]+", "_", Path(filename or "upload").stem) or "upload"
        name = f"{stem}.txt"
        n = 2
        while name in self.files:
            name = f"{stem}_{n}.txt"
            n += 1
        return name

    def add(self, filename: str, text: str) -> None:
        name = self._unique_name(filename)
        (self.folder / name).write_text(text or "", encoding="utf-8")
        self.files.append(name)

    def add_error(self, filename: str, message: str) -> None:
        self.errors.append(f"{filename}: {message}")
        (self.folder / "_errors.txt").write_text(
            "\n".join(self.errors) + "\n", encoding="utf-8"
        )

    def finish(self) -> Dict[str, Any]:
        # Zip the folder itself so it extracts as <batch_id>/<filename>.txt
        zip_path = shutil.make_archive(
            str(self.folder), "zip", root_dir=TEXT_COLLECTIONS_DIR, base_dir=self.batch_id
        )
        return {
            "batch_id": self.batch_id,
            "folder": str(self.folder),
            "zip_path": zip_path,
            "zip_url": f"/ocr/text-collections/{self.batch_id}.zip",
            "files": list(self.files),
            "errors": list(self.errors),
        }


def _shape_response(
    filename: str,
    raw_json: Dict[str, Any],
    storage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the API response.

    formatted_text  - zone-tagged, layout-aware human-readable text
    plain_text      - flat unformatted text (all pages joined)
    tables          - all tables extracted across all pages
    layout_summary  - per-page block type counts and column count
    raw_json        - full OCR output with bboxes and confidence scores
    storage         - persisted original + output folder paths (frontend uploads)
    """
    try:
        parser_text = FAST_PARSER.process_data(raw_json)
        parser_error = None
    except Exception as e:
        parser_text = ""
        parser_error = f"{type(e).__name__}: {e}"

    # Collect tables and layout summary across all pages
    all_tables = []
    layout_summary = []
    for page in raw_json.get("pages", []):
        page_tables = page.get("tables", [])
        all_tables.extend(page_tables)
        block_types: Dict[str, int] = {}
        for b in page.get("blocks", []):
            bt = b.get("block_type", "unknown")
            block_types[bt] = block_types.get(bt, 0) + 1
        layout_summary.append({
            "page": page.get("page_index", 0) + 1,
            "columns": len(page.get("columns", [])),
            "block_counts": block_types,
            "table_count": len(page_tables),
        })

    response: Dict[str, Any] = {
        "filename": filename,
        "formatted_text": parser_text,
        "plain_text": raw_json.get("text", ""),
        "tables": all_tables,
        "layout_summary": layout_summary,
        "raw_json": raw_json,
    }
    if storage:
        response["storage"] = storage
        response["document_id"] = storage.get("document_id")
        response["layout_html_url"] = storage.get("layout_html_url")
    if parser_error:
        response["parser_error"] = parser_error
    return response


def _cleaned_text(response: Dict[str, Any]) -> str:
    """Same text the frontend shows: plain_text, falling back to formatted_text."""
    for key in ("plain_text", "formatted_text"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/ocr/outputs/{document_id}/layout.html")
def download_layout_html(document_id: str):
    """Serve layout.html written for a frontend (or workflow) document id."""
    if not _DOC_ID_RE.match(document_id):
        raise HTTPException(status_code=400, detail="Invalid document_id.")
    path = OUTPUTS_DIR / document_id / "layout.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="layout.html not found for this document.")
    return FileResponse(
        path,
        media_type="text/html; charset=utf-8",
        filename=f"{document_id}_layout.html",
    )


@app.get("/ocr/text-collections/{batch_id}.zip")
def download_text_collection(batch_id: str):
    """Serve the zip of per-document .txt files written for one request."""
    if not _BATCH_ID_RE.match(batch_id):
        raise HTTPException(status_code=400, detail="Invalid batch_id.")
    path = TEXT_COLLECTIONS_DIR / f"{batch_id}.zip"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Text collection not found.")
    return FileResponse(path, media_type="application/zip", filename=f"{batch_id}.zip")


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    preset: str = Query(default="auto", description="Preprocessing preset: auto | clean_doc | photo | low_light | scan_enhance | printed_hq"),
    dpi: int = Query(default=150, ge=72, le=600, description="DPI for PDF rendering"),
    max_pages: int = Query(default=5, ge=1, le=200, description="Max pages to process"),
    return_debug: bool = Query(default=False, description="Include per-page debug info in response"),
    return_layout: bool = Query(default=True, description="Include layout zone info"),
):
    """
    Extract text from a document or image using PaddleOCR.

    Returns:
    - `formatted_text`: structured text with zone tags (HEADER, BODY, FOOTER, SIDEBAR)
    - `raw_json`: full OCR output with per-page word bounding boxes and confidence scores
    - `storage`: paths under handler_checker/storage (includes layout.html)
    """
    _check_extension(file.filename or "upload")
    suffix = Path(file.filename or "upload").suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    try:
        try:
            raw = run_ocr(
                input_path=str(tmp_path),
                preset=preset,
                dpi=dpi,
                max_pages=max_pages,
                return_debug=return_debug,
                engine="auto",
                return_layout=return_layout,
            )
            storage = _persist_frontend_upload(
                file.filename or "upload",
                tmp_path,
                raw,
            )
            response = _shape_response(file.filename or "upload", raw, storage=storage)
            collection = TextCollection()
            collection.add(file.filename or "upload", _cleaned_text(response))
            response["text_collection"] = collection.finish()
        except HTTPException:
            raise
        except Exception as e:
            # Surface OCR/processing failures as a normal JSON error response so
            # CORSMiddleware can attach headers to it. An unhandled exception here
            # falls through to Starlette's bare 500 response, which carries no
            # CORS headers and the browser reports as a generic "Failed to fetch".
            return JSONResponse(
                status_code=500,
                content={"detail": f"OCR processing failed: {type(e).__name__}: {e}"},
            )
        return JSONResponse(content=response)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/ocr-batch")
async def ocr_batch_endpoint(
    files: List[UploadFile] = File(...),
    preset: str = Query(default="auto"),
    dpi: int = Query(default=150, ge=72, le=600),
    max_pages: int = Query(default=5, ge=1, le=200),
    return_debug: bool = Query(default=False),
    return_layout: bool = Query(default=True),
):
    """
    Extract text from up to 1000 files in one request.

    Returns a list of results in the same order as the uploaded files.
    Each result includes the same storage paths as POST /ocr.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 1000:
        raise HTTPException(status_code=400, detail="Batch limit is 1000 files.")

    for f in files:
        _check_extension(f.filename or "upload")

    tmp_paths: List[Path] = []
    results: List[Dict[str, Any]] = []

    try:
        try:
            for f in files:
                suffix = Path(f.filename or "upload").suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await f.read())
                    tmp_paths.append(Path(tmp.name))

            collection = TextCollection()
            for tp, f in zip(tmp_paths, files):
                name = f.filename or "upload"
                # One bad file must not throw away the rest of the batch.
                try:
                    raw = run_ocr(
                        input_path=str(tp),
                        preset=preset,
                        dpi=dpi,
                        max_pages=max_pages,
                        return_debug=return_debug,
                        engine="auto",
                        return_layout=return_layout,
                    )
                    storage = _persist_frontend_upload(name, tp, raw)
                except Exception as e:
                    error = f"OCR processing failed: {type(e).__name__}: {e}"
                    collection.add_error(name, error)
                    results.append({"filename": name, "error": error})
                    continue
                result = _shape_response(name, raw, storage=storage)
                # Store this document's text now, before the next file starts.
                collection.add(name, _cleaned_text(result))
                results.append(result)
            text_collection = collection.finish()
        except HTTPException:
            raise
        except Exception as e:
            # See POST /ocr: an unhandled exception here would otherwise fall
            # through to Starlette's bare 500 response, which carries no CORS
            # headers and the browser reports as a generic "Failed to fetch".
            return JSONResponse(
                status_code=500,
                content={"detail": f"OCR processing failed: {type(e).__name__}: {e}"},
            )

        return JSONResponse(
            content={
                "count": len(results),
                "results": results,
                "text_collection": text_collection,
            }
        )
    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
