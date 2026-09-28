from handler_checker.models import Document
from handler_checker.metadata_manager import MetadataManager
from tests.sample_data import get_sample_pdf, reset_workspace

print("=" * 60)
print("METADATA TEST")
print("=" * 60)

reset_workspace()

SAMPLE_FILE = get_sample_pdf()
print("Sample PDF :", SAMPLE_FILE.name)

metadata = MetadataManager()

doc = Document(
    document_id="TEST_METADATA",
    filename=SAMPLE_FILE.name,
    extension=SAMPLE_FILE.suffix,
    document_type="PDF",
    original_path=str(SAMPLE_FILE),
)

metadata.add_document(doc)

assert metadata.document_exists("TEST_METADATA")
assert metadata.get_document("TEST_METADATA") is not None
assert metadata.count() == 1

print()
print("Documents:", metadata.count())
print(metadata.get_document("TEST_METADATA"))

metadata.delete_document("TEST_METADATA")
assert metadata.count() == 0

print()
print("=" * 60)
print("TEST PASSED")
print("=" * 60)
