from pathlib import Path
from typing import Iterable, List, Union

from handler_checker.handler import Handler
from handler_checker.checker import Checker
from handler_checker.retry_manager import RetryManager
from handler_checker.models import Document


class MainOrchestrator:
    """
    Coordinates the complete OCR processing workflow.
    """

    def __init__(self):
        self.handler = Handler()
        self.checker = Checker()
        self.retry_manager = RetryManager()

    def process(self, file_path: Union[str, Path]) -> Document:
        """
        Execute the complete OCR workflow for one document.
        """

        # Step 1: Upload, store, run OCR, update queue/metadata
        document, ocr_result = self.handler.process_upload(file_path)

        # Step 2: Validate using OCR output (no PDF re-read)
        self.checker.validate(document, ocr_result=ocr_result)

        # Step 3: Retry decision / completion
        self.retry_manager.retry(document)

        # Sync BOTH metadata files with final COMPLETED/FAILED state
        self.handler.metadata_manager.update_document(document)

        # Update queue status
        self.handler.queue_manager.update_status(
            document.document_id,
            "completed" if document.validation_passed else document.status.lower(),
        )

        return document

    def process_batch(
        self,
        file_paths: Iterable[Union[str, Path]],
    ) -> List[Document]:
        """
        Process multiple documents in queue order.
        """

        documents: List[Document] = []

        for file_path in file_paths:
            documents.append(self.process(file_path))

        return documents
