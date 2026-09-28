"""
=========================================================
OCR Handler + Validation Manager
Document Router
=========================================================

Routes documents to the correct processing pipeline.
"""


class DocumentRouter:
    """
    Determines which processing pipeline a document should use.
    """

    PDF_EXTENSIONS = {".pdf"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    PARSER_EXTENSIONS = {".docx", ".xlsx", ".csv", ".txt"}

    def route_document(self, document) -> str:
        """
        Select a pipeline for the document and update document.pipeline.
        """

        extension = document.extension.lower()

        if extension in self.PDF_EXTENSIONS:
            pipeline = "PDF_OCR"

        elif extension in self.IMAGE_EXTENSIONS:
            pipeline = "IMAGE_OCR"

        elif extension in self.PARSER_EXTENSIONS:
            pipeline = "PARSER"

        else:
            raise ValueError(f"Unsupported document type: {extension}")

        document.pipeline = pipeline
        return pipeline
