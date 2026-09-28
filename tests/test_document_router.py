from handler_checker.document_router import DocumentRouter
from handler_checker.models import Document
from tests.sample_data import get_sample_pdf

print("=" * 60)
print("DOCUMENT ROUTER TEST")
print("=" * 60)

router = DocumentRouter()
sample = get_sample_pdf()
print("Sample file :", sample.name)

ext = sample.suffix.lower()
expected = "PDF_OCR" if ext == ".pdf" else "IMAGE_OCR"
doc_type = "PDF" if ext == ".pdf" else "IMAGE"

document = Document(
    document_id="TEST_ROUTER",
    filename=sample.name,
    extension=ext,
    document_type=doc_type,
    original_path=str(sample),
)

pipeline = router.route_document(document)

assert pipeline == expected
assert document.pipeline == expected

print()
print("Pipeline :", pipeline)

print()
print("=" * 60)
print("TEST PASSED")
print("=" * 60)
