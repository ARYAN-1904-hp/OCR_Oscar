"""
Field extraction over OCR formatted_text.txt output.

field_extractor.py is the regex / bounding-box claims extraction engine,
kept verbatim so updated copies can be dropped in as-is. service.py is the
thin wrapper the OCR pipeline calls.
"""

from extraction.service import (
    EXCEL_FILENAME,
    extract_formatted_text,
    results_to_json,
    write_workbook,
)

__all__ = [
    "EXCEL_FILENAME",
    "extract_formatted_text",
    "results_to_json",
    "write_workbook",
]
