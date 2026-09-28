"""
=========================================================
OCR Handler + Validation Manager
Document Models
=========================================================
"""

from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from typing import Optional, Dict, Any


# ==========================================================
# Document Model
# ==========================================================

@dataclass
class Document:

    # -------------------------
    # Identity
    # -------------------------

    document_id: str
    filename: str
    extension: str
    document_type: str

    # -------------------------
    # Paths
    # -------------------------

    original_path: str
    storage_path: str = ""
    output_path: str = ""

    # -------------------------
    # Page Information
    # -------------------------

    original_pages: int = 0
    processed_pages: int = 0

    # -------------------------
    # Processing
    # -------------------------

    pipeline: str = ""
    status: str = "UPLOADED"

    retry_count: int = 0

    processing_time: float = 0.0

    upload_time: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    completed_time: Optional[str] = None

    # -------------------------
    # OCR
    # -------------------------

    ocr_status: str = ""
    ocr_engine: str = ""
    ocr_pages: int = 0
    ocr_text_length: int = 0

    # -------------------------
    # Validation
    # -------------------------

    validation_passed: bool = False

    error_message: str = ""

    # -------------------------
    # Utility Methods
    # -------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Document object into dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):

        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})

    def mark_completed(self):

        self.status = "COMPLETED"

        self.completed_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def mark_failed(self, reason: str):

        self.status = "FAILED"

        self.error_message = reason

    def increment_retry(self):

        self.retry_count += 1

    def update_processed_pages(self, pages: int):

        self.processed_pages = pages

    def update_processing_time(self, seconds: float):

        self.processing_time = round(seconds, 3)

    def validate(self):

        self.validation_passed = (
            self.original_pages == self.processed_pages
        )

        return self.validation_passed