from handler_checker.queue_manager import QueueManager
from tests.sample_data import reset_workspace

print("=" * 50)
print("QUEUE TEST")
print("=" * 50)

reset_workspace()

queue = QueueManager()
queue.clear()

queue.enqueue("DOC_A", status="uploaded")
queue.enqueue("DOC_B", status="stored")
queue.enqueue("DOC_C", status="ocr_running")

queue.update_status("DOC_C", "ocr_completed")
queue.update_status("DOC_C", "ready_for_checker")

print(queue.list_queue())
print("Queue Size:", queue.size())
print("Peek:", queue.peek())
print("DOC_C Status:", queue.get_status("DOC_C"))
print("Dequeue:", queue.dequeue())
print("Remaining:", queue.list_queue())

assert queue.get_status("DOC_C") == "ready_for_checker"
assert "DOC_B" in queue.get_queue()

queue.clear()

print()
print("=" * 50)
print("TEST PASSED")
print("=" * 50)
