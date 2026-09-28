"""
Full clean workflow for EVERY document in tests/data (PDF + images).

Usage:
    python run.py workflow
    python run.py workflow_fast
"""

import json
import os
from pathlib import Path

from handler_checker.config import ORIGINALS_DIR, OUTPUTS_DIR, METADATA_FILE, QUEUE_FILE
from handler_checker.main import MainOrchestrator
from tests.sample_data import get_sample_documents, reset_workspace

FAST = os.getenv("WORKFLOW_FAST", "").strip().lower() in {"1", "true", "yes"}

print("=" * 70)
print(
    "BATCH WORKFLOW — FAST (PDFs only)"
    if FAST
    else "BATCH WORKFLOW — ALL DOCUMENTS IN tests/data"
)
print("=" * 70)

reset_workspace()

documents_in = get_sample_documents()

if FAST:
    documents_in = [p for p in documents_in if p.suffix.lower() == ".pdf"]
    if not documents_in:
        raise FileNotFoundError("WORKFLOW_FAST=1 but no PDFs found in tests/data.")
else:
    documents_in = sorted(
        documents_in,
        key=lambda p: (
            0 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} else 1,
            p.name.lower(),
        ),
    )

print(f"\nFound {len(documents_in)} document(s):\n")
for index, path in enumerate(documents_in, start=1):
    print(f"  {index}. {path.name}")

orchestrator = MainOrchestrator()
documents = orchestrator.process_batch(documents_in)

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

expected_files = {
        "metadata.json",
        "raw_output.json",
        "formatted_text.txt",
        "layout.html",
    }
global_meta = json.loads(METADATA_FILE.read_text(encoding="utf-8") or "{}")
queue = json.loads(QUEUE_FILE.read_text(encoding="utf-8") or "[]")

assert len(documents) == len(documents_in), "Not all documents were processed."

for document in documents:
    print()
    print(f"Document ID : {document.document_id}")
    print(f"Filename    : {document.filename}")
    print(f"Pages       : {document.original_pages} / OCR {document.ocr_pages}")
    print(f"Pipeline    : {document.pipeline}")
    print(f"Engine      : {document.ocr_engine}")
    print(f"Status      : {document.status}")
    print(f"Validation  : {document.validation_passed}")
    print(f"Output      : {document.output_path}")

    assert document.original_pages >= 1
    assert document.processed_pages == document.original_pages
    assert document.ocr_pages == document.original_pages
    assert document.validation_passed is True
    assert document.status == "COMPLETED"
    assert document.completed_time is not None
    assert document.pipeline in ("PDF_OCR", "IMAGE_OCR")

    output_dir = Path(document.output_path)
    assert output_dir.exists()
    actual_files = {p.name for p in output_dir.iterdir() if p.is_file()}
    missing = expected_files - actual_files
    assert not missing, f"Missing files for {document.document_id}: {missing}"

    local_meta = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert document.document_id in global_meta
    assert global_meta[document.document_id] == local_meta
    assert local_meta["status"] == "COMPLETED"
    assert local_meta["validation_passed"] is True

for document in documents:
    output_dir = Path(document.output_path)
    assert (output_dir / "formatted_text.txt").is_file()
    assert (output_dir / "raw_output.json").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "layout.html").is_file()
    assert document.storage_path and document.document_id in Path(document.storage_path).name

output_folders = sorted(p.name for p in OUTPUTS_DIR.iterdir() if p.is_dir())
original_files = sorted(p.name for p in ORIGINALS_DIR.iterdir() if p.is_file())

assert len(queue) == len(documents)
queue_ids = [item["document_id"] for item in queue]
assert queue_ids == [doc.document_id for doc in documents]
assert all(item["status"] == "completed" for item in queue)
assert set(global_meta.keys()) == {doc.document_id for doc in documents}

print()
print("=" * 70)
print("BATCH SUMMARY")
print("=" * 70)
print(f"Documents processed: {len(documents)}")
print(f"Output folders     : {len(output_folders)}")
print(f"Originals stored   : {len(original_files)}")
print(f"Queue entries      : {len(queue)}")
print(f"Metadata documents : {len(global_meta)}")
print("All statuses        : COMPLETED")
print("Metadata sync       : OK")
print()
print("=" * 70)
print("TEST PASSED")
print("=" * 70)
