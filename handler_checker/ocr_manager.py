"""
OCR Manager

Acts as the integration layer between the Handler and the OCR Engine.

Responsibilities:
- Receive document path from Handler
- Call OCR pipeline
- Return OCR results
- Keep Handler independent of OCR implementation
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from ocr.pipeline import run_ocr
from ocr.html_renderer import render_layout_html
from ocr.regex_ready import build_regex_ready_text
from extraction import EXCEL_FILENAME, extract_formatted_text, write_workbook
from handler_checker.logger import logger


class OCRManager:
    """
    Bridge between Handler and OCR Engine.
    """

    def __init__(self):
        pass

    def process_document(
        self,
        document_path: str,
        output_dir: Optional[str] = None,
        preset: str = "clean_doc",
        dpi: int = 120,
        max_pages: int = 200,
        return_layout: bool = True,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a document using the OCR engine.

        Parameters
        ----------
        document_path : str
            Path to the input document.
        output_dir : str, optional
            Folder where OCR outputs should be saved.

        Returns
        -------
        Dict[str, Any]
            OCR output containing:
            - extracted text
            - pages
            - metadata
        """

        if not Path(document_path).exists():
            raise FileNotFoundError(f"Document not found: {document_path}")

        result = run_ocr(
            input_path=document_path,
            preset=preset,
            dpi=dpi,
            max_pages=max_pages,
            return_debug=return_debug,
            return_layout=return_layout,
        )

        if output_dir:
            self.save_ocr_output(
                result,
                output_dir,
                source_path=document_path,
            )

        return result

    def save_ocr_output(
        self,
        ocr_result: Dict[str, Any],
        output_dir: str,
        source_path: Optional[str] = None,
        source_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Persist OCR outputs:
          - raw_output.json
          - formatted_text.txt     (single regex-ready combo text)
          - layout.html
          - extracted_fields.xlsx  (regex field extraction over formatted_text.txt)
        Per-doc metadata.json is written by the handler/metadata manager.

        Returns the extraction outcome: {"results": [(label, DocumentResult)],
        "excel_path": Path | None, "error": str | None}. An extraction
        failure never fails the OCR save.
        """

        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / "raw_output.json", "w", encoding="utf-8") as f:
            json.dump(ocr_result, f, indent=2, ensure_ascii=False)

        # Single text file — regex-ready (TEXT / LAYOUT / TABLES / BOXES / FIELDS)
        formatted_text = build_regex_ready_text(ocr_result)
        with open(folder / "formatted_text.txt", "w", encoding="utf-8") as f:
            f.write(formatted_text)

        render_layout_html(ocr_result, folder, source_path=source_path)

        label = source_label or (Path(source_path).name if source_path else folder.name)
        return self.extract_fields(formatted_text, folder, label)

    def extract_fields(
        self,
        formatted_text: str,
        output_dir: Path,
        source_label: str,
    ) -> Dict[str, Any]:
        """Run the regex field extractor and write extracted_fields.xlsx."""
        outcome: Dict[str, Any] = {"results": [], "excel_path": None, "error": None}
        try:
            results = extract_formatted_text(formatted_text, source_label)
            outcome["results"] = results
            if results:
                outcome["excel_path"] = write_workbook(
                    results, Path(output_dir) / EXCEL_FILENAME
                )
        except Exception as e:
            outcome["error"] = f"{type(e).__name__}: {e}"
            logger.exception("Field extraction failed for %s", source_label)
        return outcome
