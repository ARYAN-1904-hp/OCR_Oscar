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
    ) -> None:
        """
        Persist OCR outputs:
          - raw_output.json
          - formatted_text.txt     (single regex-ready combo text)
          - layout.html
        Per-doc metadata.json is written by the handler/metadata manager.
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

