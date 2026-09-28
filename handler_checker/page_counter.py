from pathlib import Path
import fitz

class PageCounter:
    """
    Counts pages for supported document formats.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    TIFF_EXTENSIONS = {".tif", ".tiff"}
    TEXT_EXTENSIONS = {".docx", ".xlsx", ".csv", ".txt"}

    def count_pages(self, document):
        """
        Count the number of pages in a document and update the Document object.
        """

        extension = document.extension.lower()
        file_path = Path(document.storage_path)

        if extension == ".pdf":
            pages = self._count_pdf(file_path)

        elif extension in self.TIFF_EXTENSIONS:
            pages = self._count_tiff(file_path)

        elif extension in self.IMAGE_EXTENSIONS:
            pages = 1

        elif extension in self.TEXT_EXTENSIONS:
            # Placeholder until parser module is integrated
            pages = 1

        else:
            raise ValueError(f"Unsupported document type: {extension}")

        document.original_pages = pages
        return pages

    def _count_pdf(self, file_path: Path) -> int:
        doc = fitz.open(str(file_path))
        n = len(doc)
        doc.close()
        return n

    def _count_tiff(self, file_path: Path) -> int:
        try:
            from PIL import Image

            image = Image.open(file_path)
            return getattr(image, "n_frames", 1)

        except Exception:
            return 1