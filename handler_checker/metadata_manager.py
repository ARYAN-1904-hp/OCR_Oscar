"""
=========================================================
OCR Handler + Validation Manager
Metadata Manager
=========================================================

Owns all metadata persistence:
  • Global: handler_checker/metadata/metadata.json
  • Local:  <document.output_path>/metadata.json

Both files are kept synchronized on every write.
"""

import json
from pathlib import Path

from .config import METADATA_FILE
from .models import Document


class MetadataManager:

    def __init__(self):

        self.metadata_file = Path(METADATA_FILE)

        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.metadata_file.exists() or self.metadata_file.stat().st_size == 0:
            self.metadata_file.write_text("{}", encoding="utf-8")

    # =====================================================

    def load(self):

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                return {}

            return json.loads(content)

        except json.JSONDecodeError:
            return {}

    # =====================================================

    def save(self, data):

        with open(self.metadata_file, "w", encoding="utf-8") as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

    # =====================================================
    # Save metadata.json inside each document output folder
    # =====================================================

    def save_document_metadata(self, document: Document):
        """
        Write local metadata.json for one document.
        """

        if not document.output_path:
            return

        output_dir = Path(document.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = output_dir / "metadata.json"

        with open(metadata_file, "w", encoding="utf-8") as f:

            json.dump(
                document.to_dict(),
                f,
                indent=4,
                ensure_ascii=False
            )

    # =====================================================

    def _sync(self, document: Document):
        """
        Persist the document to both global and local metadata.
        """

        metadata = self.load()
        metadata[document.document_id] = document.to_dict()
        self.save(metadata)
        self.save_document_metadata(document)

    # =====================================================

    def add_document(self, document: Document):

        self._sync(document)

    # =====================================================

    def update_document(self, document: Document):

        self._sync(document)

    # =====================================================

    def get_document(self, document_id):

        metadata = self.load()

        if document_id not in metadata:
            return None

        return Document.from_dict(metadata[document_id])

    # =====================================================

    def delete_document(self, document_id):

        metadata = self.load()

        if document_id in metadata:

            del metadata[document_id]

            self.save(metadata)

            return True

        return False

    # =====================================================

    def document_exists(self, document_id):

        metadata = self.load()

        return document_id in metadata

    # =====================================================

    def list_documents(self):

        metadata = self.load()

        return [
            Document.from_dict(doc)
            for doc in metadata.values()
        ]

    # =====================================================

    def count(self):

        return len(self.load())
