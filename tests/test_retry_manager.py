from handler_checker.retry_manager import RetryManager
from handler_checker.models import Document

print("=" * 60)
print("RETRY MANAGER TEST")
print("=" * 60)

retry_manager = RetryManager()

document = Document(
    document_id="TEST_RETRY",
    filename="sample.pdf",
    extension=".pdf",
    document_type="PDF",
    pipeline="PDF_OCR",
    original_path="sample.pdf",
)

document.validation_passed = False
document.retry_count = 0

result = retry_manager.retry(document)
assert result is True
assert document.retry_count == 1
assert document.status == "RETRY"
print("Retry 1 Passed")

result = retry_manager.retry(document)
assert result is True
assert document.retry_count == 2
assert document.status == "RETRY"
print("Retry 2 Passed")

result = retry_manager.retry(document)
assert result is True
assert document.retry_count == 3
assert document.status == "RETRY"
print("Retry 3 Passed")

result = retry_manager.retry(document)
assert result is False
assert document.retry_count == 3
assert document.status == "FAILED"
print("Retry Limit Passed")

# Success path should mark completed_time
ok_doc = Document(
    document_id="TEST_RETRY_OK",
    filename="sample.pdf",
    extension=".pdf",
    document_type="PDF",
    pipeline="PDF_OCR",
    original_path="sample.pdf",
)
ok_doc.validation_passed = True
assert retry_manager.retry(ok_doc) is False
assert ok_doc.status == "COMPLETED"
assert ok_doc.completed_time is not None
print("Completed Path Passed")

print()
print("=" * 60)
print("TEST PASSED")
print("=" * 60)
