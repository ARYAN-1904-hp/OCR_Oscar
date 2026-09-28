from handler_checker.ocr_manager import OCRManager
from tests.sample_data import get_sample_pdf

print("=" * 60)
print("OCR MANAGER TEST")
print("=" * 60)

sample = get_sample_pdf()
print("Sample PDF :", sample.name)

ocr = OCRManager()

# No output_dir → no extra folders/files written
result = ocr.process_document(
    document_path=str(sample),
    max_pages=2,
)

assert "text" in result, "OCR result missing 'text' key."
assert len(result.get("pages", [])) >= 1

print()
print("Keys :", list(result.keys()))
print()
print("Text preview:")
print((result.get("text") or "")[:1000])

print()
print("=" * 60)
print("TEST PASSED")
print("=" * 60)
