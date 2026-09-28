from handler_checker.models import Document
from handler_checker.storage_manager import StorageManager
from handler_checker.page_counter import PageCounter
from tests.sample_data import get_sample_pdf, reset_workspace

print("=" * 60)
print("PAGE COUNTER TEST")
print("=" * 60)

reset_workspace()

sample = get_sample_pdf()
print("Sample PDF :", sample.name)

document = Document(
    document_id="TEST_PAGE_COUNTER",
    filename=sample.name,
    extension=sample.suffix,
    document_type="PDF",
    original_path=str(sample),
)

storage = StorageManager()
storage.save_original(document)

counter = PageCounter()
pages = counter.count_pages(document)

assert pages >= 1
assert document.original_pages == pages

print()
print("Pages :", pages)

# Do not leave storage junk behind
storage.cleanup(document)

print()
print("=" * 60)
print("TEST PASSED")
print("=" * 60)
