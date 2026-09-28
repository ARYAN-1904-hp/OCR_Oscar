"""
Glue between the OCR pipeline and the regex field extractor.

    formatted_text.txt  ->  extract_all_documents()  ->  extracted_fields.xlsx

formatted_text.txt (ocr/regex_ready.py) already carries everything the
extractor parses: the "# OCR meta | engine=" document header, [BOXES]
LINE/WORD bounding boxes, and [TABLE] CELL[r,c] lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from extraction.field_extractor import (
    DocumentResult,
    build_dashboard_workbook,
    extract_all_documents,
)

EXCEL_FILENAME = "extracted_fields.xlsx"

# (source label, result) -- the source label becomes the "Source File" /
# "Screenshot" column in every workbook sheet.
LabeledResult = Tuple[str, DocumentResult]


def extract_formatted_text(formatted_text: str, source_label: str) -> List[LabeledResult]:
    """Run the extractor over one formatted_text dump."""
    return [(source_label, r) for r in extract_all_documents(formatted_text)]


def write_workbook(results: List[LabeledResult], output_path: Path) -> Path:
    """Write the 6-sheet dashboard workbook (Dataset A/B, Scenario Summary,
    Consolidated, Raw Extracted, Data Dictionary).

    Pass results from several files together to get cross-screenshot
    scenario linking (shared Ticket / Claim / OSC / Member IDs)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_dashboard_workbook(results, str(output_path))
    return output_path


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def results_to_json(results: List[LabeledResult]) -> List[Dict[str, Any]]:
    """API-friendly view of the per-document extraction."""
    return [
        {
            "source": source,
            "document_index": r.document_index,
            "detected_application": r.detected_application,
            "fields": {k: _jsonable(v) for k, v in r.fields.items()},
            "extra_fields": {k: _jsonable(v) for k, v in r.extra_fields.items()},
            "flags": dict(r.flags),
        }
        for source, r in results
    ]
