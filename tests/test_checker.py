import tempfile
from pathlib import Path

from handler_checker.checker import Checker
from handler_checker.models import Document

print("=" * 60)
print("CHECKER TEST")
print("=" * 60)

checker = Checker()

with tempfile.TemporaryDirectory() as tmp:
    output_directory = Path(tmp)

    (output_directory / "formatted_text.txt").write_text(
        "Sample OCR extracted text for validation.",
        encoding="utf-8",
    )

    document = Document(
        document_id="TEST_CHECKER",
        filename="sample.pdf",
        extension=".pdf",
        document_type="PDF",
        pipeline="PDF_OCR",
        original_path="sample.pdf",
        output_path=str(output_directory),
    )

    document.original_pages = 5
    document.processed_pages = 5
    document.ocr_pages = 5

    ocr_result = {
        "text": "Sample OCR extracted text for validation.",
        "pages": [{"page": i} for i in range(1, 6)],
        "meta": {"engine": "paddleocr"},
    }

    result = checker.validate(document, ocr_result=ocr_result)
    assert result is True

    print()
    print("Validation Passed :", result)
    print("Page Validation   :", checker.validate_pages(document, ocr_result))
    print("OCR Text Validation:", checker.validate_ocr_text(document, ocr_result))

print()
print("=" * 60)
print("TEST PASSED")
print("=" * 60)
