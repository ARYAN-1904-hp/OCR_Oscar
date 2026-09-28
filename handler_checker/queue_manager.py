"""
=========================================================
OCR Handler + Validation Manager
Queue Manager
=========================================================

Tracks documents and their processing status:

Uploaded → Stored → OCR Running → OCR Completed → Ready For Checker
"""

import json
from pathlib import Path

from .config import QUEUE_FILE


class QueueManager:

    def __init__(self):

        self.queue_file = Path(QUEUE_FILE)

        self.queue_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.queue_file.exists() or self.queue_file.stat().st_size == 0:
            self.queue_file.write_text("[]", encoding="utf-8")

    # =====================================================

    def _normalize(self, queue):
        """
        Convert legacy string entries into status objects.
        """

        normalized = []

        for item in queue:
            if isinstance(item, str):
                normalized.append(
                    {
                        "document_id": item,
                        "status": "uploaded",
                    }
                )
            elif isinstance(item, dict) and "document_id" in item:
                normalized.append(
                    {
                        "document_id": item["document_id"],
                        "status": item.get("status", "uploaded"),
                    }
                )

        return normalized

    # =====================================================

    def load(self):

        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                return []

            return self._normalize(json.loads(content))

        except json.JSONDecodeError:
            return []

    # =====================================================

    def save(self, queue):

        with open(self.queue_file, "w", encoding="utf-8") as f:
            json.dump(self._normalize(queue), f, indent=4)

    # =====================================================

    def enqueue(self, document_id, status="uploaded"):

        queue = self.load()

        for item in queue:
            if item["document_id"] == document_id:
                item["status"] = status
                self.save(queue)
                return

        queue.append(
            {
                "document_id": document_id,
                "status": status,
            }
        )

        self.save(queue)

    # =====================================================

    def update_status(self, document_id, status):

        queue = self.load()

        for item in queue:
            if item["document_id"] == document_id:
                item["status"] = status
                self.save(queue)
                return True

        # If not present yet, add it
        queue.append(
            {
                "document_id": document_id,
                "status": status,
            }
        )
        self.save(queue)
        return True

    # =====================================================

    def get_status(self, document_id):

        for item in self.load():
            if item["document_id"] == document_id:
                return item["status"]

        return None

    # =====================================================

    def dequeue(self):

        queue = self.load()

        if len(queue) == 0:
            return None

        document = queue.pop(0)

        self.save(queue)

        return document["document_id"]

    # =====================================================

    def peek(self):

        queue = self.load()

        if len(queue) == 0:
            return None

        return queue[0]["document_id"]

    # =====================================================

    def remove(self, document_id):

        queue = self.load()
        original_size = len(queue)

        queue = [
            item for item in queue
            if item["document_id"] != document_id
        ]

        if len(queue) != original_size:
            self.save(queue)
            return True

        return False

    # =====================================================

    def clear(self):

        self.save([])

    # =====================================================

    def size(self):

        return len(self.load())

    # =====================================================

    def is_empty(self):

        return self.size() == 0

    # =====================================================

    def list_queue(self):
        """
        Return full queue entries with status.
        """

        return self.load()

    # =====================================================

    def get_queue(self):
        """
        Return document IDs currently in the queue.
        """

        return [item["document_id"] for item in self.load()]
