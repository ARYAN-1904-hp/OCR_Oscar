"""
=========================================================
OCR Handler & Validation Manager
Integration Test
=========================================================
"""

from handler_checker.models import Document
from handler_checker.storage_manager import StorageManager
from handler_checker.metadata_manager import MetadataManager
from handler_checker.queue_manager import QueueManager
from tests.sample_data import get_sample_pdf, reset_workspace

print("=" * 80)
print("OCR HANDLER COMPONENT - INTEGRATION TEST")
print("=" * 80)

reset_workspace()

storage = StorageManager()
metadata = MetadataManager()
queue = QueueManager()

sample_file = get_sample_pdf()
print("Sample PDF :", sample_file.name)

document = Document(
    document_id="TEST_INTEGRATION",
    filename=sample_file.name,
    extension=sample_file.suffix,
    document_type="PDF",
    original_path=str(sample_file.resolve()),
)

print("\n[1] Document Created")
print(document)

storage.save_original(document)
storage.create_output_directory(document)

print("\n[2] Original Stored")
print(document.storage_path)

metadata.add_document(document)
print("\n[3] Metadata Stored")

queue.enqueue(document.document_id, status="uploaded")
print("\n[4] Added To Queue")

loaded = metadata.get_document(document.document_id)
print("\n[5] Metadata Verification")
print(loaded)

print("\n[6] Queue Verification")
print(queue.list_queue())

print("\n[7] Storage Verification")
print("Storage Exists :", storage.document_exists(document))
print("Output Folder  :", document.output_path)

assert storage.document_exists(document)
assert metadata.document_exists(document.document_id)
assert document.document_id in queue.get_queue()

# Cleanup so full suite does not leave junk
storage.cleanup(document)
metadata.delete_document(document.document_id)
queue.clear()

print("\n" + "=" * 80)
print("INTEGRATION TEST SUMMARY")
print("=" * 80)
print(f"Document ID      : {document.document_id}")
print(f"Filename         : {document.filename}")
print("Cleanup          : OK")
print("=" * 80)
print("ALL INTEGRATION TESTS PASSED")
print("=" * 80)
