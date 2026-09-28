from handler_checker.models import Document
from handler_checker.storage_manager import StorageManager
from tests.sample_data import get_sample_pdf, reset_workspace

print("=" * 60)
print("STORAGE TEST")
print("=" * 60)

reset_workspace()

SAMPLE_FILE = get_sample_pdf()
print("Sample PDF :", SAMPLE_FILE.name)

doc = Document(
    document_id="TEST_STORAGE",
    filename=SAMPLE_FILE.name,
    extension=SAMPLE_FILE.suffix,
    document_type="PDF",
    original_path=str(SAMPLE_FILE),
)

storage = StorageManager()

print()
print("Saving Original...")
storage.save_original(doc)
print(doc.storage_path)

print()
print("Creating Output Directory...")
storage.create_output_directory(doc)
print(doc.output_path)

print()
print("JSON Path")
print(storage.get_json_output_path(doc))

assert doc.storage_path
assert doc.output_path

storage.cleanup(doc)

print()
print("=" * 60)
print("TEST PASSED")
print("=" * 60)
