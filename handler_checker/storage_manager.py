"""
=========================================================
OCR Handler + Validation Manager
Storage Manager
=========================================================
"""

from pathlib import Path
import shutil

from .config import ORIGINALS_DIR, OUTPUTS_DIR
from .models import Document


class StorageManager:
    """
    Handles all storage-related operations.

    Responsibilities:
        • Store original uploaded documents
        • Create output folders
        • Delete files/folders
        • Check file existence
    """

    def __init__(self):
        ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # Save Original Document
    # =====================================================

    def save_original(self, document: Document) -> str:
        """
        Copies uploaded file into storage/originals.

        Example:
            D0001_claim.pdf
        """

        source = Path(document.original_path)

        filename = f"{document.document_id}_{source.name}"

        destination = ORIGINALS_DIR / filename

        shutil.copy2(source, destination)

        document.storage_path = str(destination)

        return str(destination)

    # =====================================================
    # Create Output Folder
    # =====================================================

    def create_output_directory(self, document: Document) -> str:
        """
        Creates

        storage/
            outputs/
                D0001/
        """

        output_folder = OUTPUTS_DIR / document.document_id

        output_folder.mkdir(parents=True, exist_ok=True)

        document.output_path = str(output_folder)

        return str(output_folder)

    # =====================================================
    # Check Document Exists
    # =====================================================

    def document_exists(self, document: Document) -> bool:

        return Path(document.storage_path).exists()

    # =====================================================
    # Delete File
    # =====================================================

    def delete_document(self, document: Document) -> bool:
        if not document.storage_path or not document.storage_path.strip():
            return False
        path = Path(document.storage_path)
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False

    # =====================================================
    # Delete Output Folder
    # =====================================================

    def delete_output_directory(self, document: Document):
        if not document.output_path or not document.output_path.strip():
            return
        folder = Path(document.output_path)
        if folder.exists() and folder.is_dir() and folder.resolve() != Path(".").resolve():
            shutil.rmtree(folder)

    # =====================================================
    # Cleanup
    # =====================================================

    def cleanup(self, document: Document):

        self.delete_document(document)

        self.delete_output_directory(document)

    # =====================================================
    # Clear all stored originals / outputs
    # =====================================================

    def clear_all(self):

        if ORIGINALS_DIR.exists():
            shutil.rmtree(ORIGINALS_DIR)
        if OUTPUTS_DIR.exists():
            shutil.rmtree(OUTPUTS_DIR)

        ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # Get JSON Output File
    # =====================================================

    def get_json_output_path(self, document: Document) -> str:

        return str(
            Path(document.output_path)
            / f"{document.document_id}.json"
        )

    # =====================================================
    # Get Log File
    # =====================================================

    def get_log_path(self, document: Document) -> str:

        return str(
            Path(document.output_path)
            / "processing.log"
        )