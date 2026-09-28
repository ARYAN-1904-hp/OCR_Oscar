from pathlib import Path
from typing import Any, Dict, Optional


class Checker:
    """
    Validates OCR outputs before marking the document
    as successfully processed.

    Uses saved OCR output (formatted_text.txt / ocr_result)
    instead of re-opening the original PDF.
    """

    def __init__(self):
        pass

    def validate_pages(self, document, ocr_result: Optional[Dict[str, Any]] = None):
        """
        Validate whether all pages were processed.
        Prefer OCR result page count when available.
        """

        if ocr_result is not None:
            ocr_pages = len(ocr_result.get("pages", []))
            if ocr_pages > 0:
                return document.original_pages == ocr_pages

        if document.ocr_pages > 0:
            return document.original_pages == document.ocr_pages

        return document.original_pages == document.processed_pages

    def validate_output_directory(self, document):
        """
        Validate output directory exists.
        """

        return bool(document.output_path) and Path(document.output_path).exists()

    def validate_ocr_text(
        self,
        document,
        ocr_result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Validate OCR text from memory or formatted_text.txt.
        """

        text = ""

        if ocr_result is not None:
            text = ocr_result.get("text", "") or ""
        else:
            text_file = Path(document.output_path) / "formatted_text.txt"

            if not text_file.exists():
                return False

            text = text_file.read_text(encoding="utf-8")

        return len(text.strip()) > 0

    def validate(
        self,
        document,
        ocr_result: Optional[Dict[str, Any]] = None,
    ):
        """
        Execute all validations using OCR output and
        update document.validation_passed.
        """

        passed = (
            self.validate_output_directory(document)
            and self.validate_pages(document, ocr_result)
            and self.validate_ocr_text(document, ocr_result)
        )

        document.validation_passed = passed
        return passed
