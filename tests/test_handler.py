from pathlib import Path

from handler_checker.handler import Handler
from tests.sample_data import get_sample_pdf, reset_workspace

print("=" * 60)
print("HANDLER TEST")
print("=" * 60)

reset_workspace()

handler = Handler()
sample = get_sample_pdf()
print("Sample file :", sample.name)

expected_pipeline = (
    "PDF_OCR" if sample.suffix.lower() == ".pdf" else "IMAGE_OCR"
)

document, ocr_result = handler.process_upload(sample)

assert Path(document.storage_path).exists(), "Original file was not stored."
assert Path(document.output_path).exists(), "Output directory was not created."
assert (Path(document.output_path) / "formatted_text.txt").exists()
assert (Path(document.output_path) / "raw_output.json").exists()
assert (Path(document.output_path) / "metadata.json").exists()
assert (Path(document.output_path) / "layout.html").exists()
stored_document = handler.metadata_manager.get_document(document.document_id)
assert stored_document is not None
assert stored_document.original_pages == document.original_pages
assert stored_document.pipeline == document.pipeline
assert stored_document.ocr_status == "completed"

assert document.document_id in handler.queue_manager.get_queue()
assert handler.queue_manager.get_status(document.document_id) == "ready_for_checker"
assert document.status == "READY_FOR_CHECKER"
assert document.original_pages >= 1
assert document.pipeline == expected_pipeline
assert ocr_result is not None
assert document.ocr_pages == document.original_pages

print()
print("=" * 60)
print("DOCUMENT CREATED")
print("=" * 60)
print(document)

print()
print("=" * 60)
print("OUTPUT FILES")
print("=" * 60)
for item in sorted(Path(document.output_path).iterdir()):
    print(" -", item.name)

print()
print("=" * 60)
print("ALL HANDLER TESTS PASSED")
print("=" * 60)
