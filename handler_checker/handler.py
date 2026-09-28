"""
Handler Module

The Handler is the entry point of the OCR Handler & Validation Manager.

Responsibilities:
- Accept uploaded documents
- Validate uploaded files
- Generate Document IDs
- Create Document objects
- Coordinate storage
- Create metadata
- Run OCR
- Update OCR metadata / queue status
- Save OCR outputs
- Mark documents ready for Checker
"""

import time
from datetime import datetime
from pathlib import Path

from handler_checker.logger import logger
from handler_checker.config import SUPPORTED_EXTENSIONS
from handler_checker.models import Document
from handler_checker.storage_manager import StorageManager
from handler_checker.metadata_manager import MetadataManager
from handler_checker.queue_manager import QueueManager
from handler_checker.page_counter import PageCounter
from handler_checker.document_router import DocumentRouter
from handler_checker.checker import Checker
from handler_checker.retry_manager import RetryManager
from handler_checker.ocr_manager import OCRManager


class Handler:
    """
    Main document handler.

    Coordinates preprocessing and OCR for uploaded documents.
    """

    def __init__(self):
        """
        Initialize all required managers.
        """

        self.storage_manager = StorageManager()
        self.metadata_manager = MetadataManager()
        self.queue_manager = QueueManager()
        self.page_counter = PageCounter()
        self.document_router = DocumentRouter()
        self.checker = Checker()
        self.retry_manager = RetryManager()
        self.ocr_manager = OCRManager()

        logger.info("Handler initialized successfully.")

    def _generate_document_id(self) -> str:
        """
        Generate a unique document ID (safe for batch runs).
        """

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

        return f"DOC_{timestamp}"

    def _get_document_type(self, extension: str) -> str:
        """
        Determine document type based on extension.
        """

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
        }

        parser_extensions = {
            ".docx",
            ".xlsx",
            ".csv",
            ".txt",
        }

        if extension in image_extensions:
            return "IMAGE"

        if extension in parser_extensions:
            return "PARSER"

        if extension == ".pdf":
            return "PDF"

        return "UNKNOWN"

    def _update_ocr_metadata(self, document: Document, ocr_result: dict) -> None:
        """
        Enrich document metadata with OCR details.
        """

        meta = ocr_result.get("meta") or {}
        document.ocr_status = "completed"

        # Prefer real engine used; mark digital-text extraction clearly
        if meta.get("digital_pdf"):
            document.ocr_engine = meta.get("engine_used") or "digital_pdf+layout"
        else:
            base_engine = (
                meta.get("engine_used")
                or meta.get("engine")
                or "paddleocr"
            )
            if meta.get("ppstructure") or meta.get("layout_engine") == "HybridLayoutEngine":
                # Only label +ppstructure when it was actually enabled
                if meta.get("ppstructure"):
                    document.ocr_engine = f"{base_engine}+ppstructure"
                else:
                    document.ocr_engine = base_engine
            else:
                document.ocr_engine = base_engine
        document.ocr_pages = len(ocr_result.get("pages", []))
        document.ocr_text_length = len(ocr_result.get("text", ""))
        document.processed_pages = document.ocr_pages

    def process_upload(self, file_path: str):
        """
        Validate uploaded document, run OCR, update queue/metadata,
        and mark the document ready for Checker.

        Returns
        -------
        tuple[Document, dict | None]
            Document object and OCR result (None if OCR failed).
        """

        try:

            file_path = Path(file_path)

            # Check file exists
            if not file_path.exists():
                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )

            # Check extension
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type: {file_path.suffix}"
                )

            logger.info("Document validation successful.")

            # Create Document object
            document = Document(
                document_id=self._generate_document_id(),
                filename=file_path.name,
                extension=file_path.suffix.lower(),
                document_type=self._get_document_type(
                    file_path.suffix.lower()
                ),
                original_path=str(file_path),
            )

            document.status = "UPLOADED"
            self.queue_manager.enqueue(document.document_id, status="uploaded")

            logger.info(
                f"[{document.document_id}] Document object created successfully."
            )

            # Store original file
            self.storage_manager.save_original(document)

            # Create output directory
            self.storage_manager.create_output_directory(document)

            # Count pages
            self.page_counter.count_pages(document)

            # Route document
            self.document_router.route_document(document)

            document.status = "STORED"
            self.queue_manager.update_status(document.document_id, "stored")
            self.metadata_manager.add_document(document)

            logger.info(
                f"[{document.document_id}] Original file stored successfully."
            )

            # -------------------------------------------------
            # Run OCR
            # -------------------------------------------------
            document.status = "OCR_RUNNING"
            self.queue_manager.update_status(document.document_id, "ocr_running")
            self.metadata_manager.update_document(document)

            ocr_result = None

            try:
                ocr_started = time.perf_counter()

                # clean_doc = single-pass preprocess (much faster than multi-pass "auto")
                ocr_preset = "clean_doc"

                ocr_result = self.ocr_manager.process_document(
                    document_path=document.storage_path,
                    output_dir=document.output_path,
                    preset=ocr_preset,
                    dpi=120,
                    # Process every page counted for this document
                    max_pages=max(document.original_pages, 1),
                )

                document.update_processing_time(
                    time.perf_counter() - ocr_started
                )

                # Update metadata with OCR details
                self._update_ocr_metadata(document, ocr_result)

                document.status = "OCR_COMPLETED"
                self.queue_manager.update_status(
                    document.document_id,
                    "ocr_completed",
                )
                self.metadata_manager.update_document(document)

                logger.info(
                    f"[{document.document_id}] OCR completed successfully."
                )

                # Ready for Checker
                document.status = "READY_FOR_CHECKER"
                self.queue_manager.update_status(
                    document.document_id,
                    "ready_for_checker",
                )

            except Exception as ocr_error:
                document.ocr_status = "failed"
                document.error_message = str(ocr_error)
                document.status = "OCR_FAILED"
                self.queue_manager.update_status(
                    document.document_id,
                    "ocr_failed",
                )

                logger.exception(
                    f"[{document.document_id}] OCR failed: {ocr_error}"
                )

            # Sync global + local metadata with final upload-stage status
            self.metadata_manager.update_document(document)

            logger.info(
                f"[{document.document_id}] Upload pipeline finished "
                f"with status={document.status}."
            )

            return document, ocr_result

        except Exception as e:

            logger.exception(f"Handler failed: {e}")

            raise
