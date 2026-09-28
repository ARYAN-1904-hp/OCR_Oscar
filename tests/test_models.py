from handler_checker.models import Document
from tests.sample_data import get_sample_pdf

SAMPLE_FILE = get_sample_pdf()
print("Sample PDF :", SAMPLE_FILE.name)

doc = Document(
    document_id="TEST_MODELS",
    filename=SAMPLE_FILE.name,
    extension=SAMPLE_FILE.suffix,
    document_type="PDF",
    original_path=str(SAMPLE_FILE),
)

print("=" * 50)
print("INITIAL DOCUMENT")
print("=" * 50)
print(doc)

doc.original_pages = 5
doc.processed_pages = 5
doc.mark_completed()
doc.processing_time = 3.84
doc.validation_passed = True

print()
print("=" * 50)
print("UPDATED DOCUMENT")
print("=" * 50)
print(doc)

print()
print("=" * 50)
print("DICTIONARY OUTPUT")
print("=" * 50)
print(doc.to_dict())

assert doc.status == "COMPLETED"
assert doc.completed_time is not None
