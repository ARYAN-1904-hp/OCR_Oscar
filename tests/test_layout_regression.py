"""
Regression test for the digital-PDF layout heuristics (ocr/digital_layout.py).

Pins the block/table output produced for a handful of real intake forms so
that future tuning of the block-type classifier or table-row assignment in
ocr/digital_layout.py can't silently change behaviour on other layouts
without a test failure calling it out.

This targets the digital-PDF path (PyMuPDF + pdfplumber) specifically
because it is deterministic and dependency-light - no PaddleOCR / PPStructure
models need to be loaded, so the test runs in seconds and its output can't
drift from run to run.

To regenerate the pinned fixtures after an intentional heuristic change, run:
    python tests/test_layout_regression.py --update-baseline
"""

import json
import sys
from pathlib import Path

import fitz
import pdfplumber

from ocr.digital_layout import build_digital_page_result
from tests.sample_data import DATA_DIR

FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "expected_layout"

# (fixture name, pdf filename, page index) - page indices were chosen because
# they contain both a real table and non-table text (headings/paragraphs),
# giving the block classifier and the table-row heuristic something to do.
SAMPLES = [
    ("chw_page0", "Filled_CA_CHW_Supplemental_Provider_Review_Form_Dummy.pdf", 0),
    ("cs_page4", "Filled_CA_CS_Credentialing_Application_Dummy (1).pdf", 4),
    ("ecm_page4", "Filled_CA_ECM_Credentialing_Application_Dummy.pdf", 4),
]


def summarize_page(pr) -> dict:
    """Reduce a PageResult to a plain-dict summary suitable for pinning.

    Captures the layout heuristics' observable behaviour (block types in
    reading order, bboxes, table structure/text) without depending on
    incidental details like Line/Word object identity.
    """
    block_types = [b.block_type for b in pr.blocks]
    counts: dict = {}
    for bt in block_types:
        counts[bt] = counts.get(bt, 0) + 1

    return {
        "num_blocks": len(pr.blocks),
        "block_type_counts": counts,
        "block_type_sequence": block_types,
        "blocks": [
            {
                "block_type": b.block_type,
                "bbox": b.bbox,
                "reading_order": b.reading_order,
                "text": "\n".join(ln.text for ln in b.lines),
            }
            for b in pr.blocks
        ],
        "tables": [
            {
                "bbox": t.get("bbox"),
                "text": t.get("text"),
            }
            for t in pr.tables
        ],
    }


def build_summary(pdf_name: str, page_index: int) -> dict:
    path = DATA_DIR / pdf_name
    if not path.exists():
        raise FileNotFoundError(f"Sample form not found: {path}")

    doc = fitz.open(str(path))
    try:
        with pdfplumber.open(str(path)) as pdf:
            pr = build_digital_page_result(
                doc.load_page(page_index), pdf.pages[page_index], page_index
            )
    finally:
        doc.close()

    return summarize_page(pr)


def load_expected(fixture_name: str) -> dict:
    fixture_path = FIXTURE_DIR / f"{fixture_name}.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def diff_summary(name: str, expected: dict, actual: dict) -> list:
    problems = []

    if expected["block_type_sequence"] != actual["block_type_sequence"]:
        problems.append(
            f"[{name}] block_type_sequence changed:\n"
            f"  expected: {expected['block_type_sequence']}\n"
            f"  actual  : {actual['block_type_sequence']}"
        )

    if expected["block_type_counts"] != actual["block_type_counts"]:
        problems.append(
            f"[{name}] block_type_counts changed:\n"
            f"  expected: {expected['block_type_counts']}\n"
            f"  actual  : {actual['block_type_counts']}"
        )

    if len(expected["tables"]) != len(actual["tables"]):
        problems.append(
            f"[{name}] table count changed: "
            f"expected {len(expected['tables'])}, got {len(actual['tables'])}"
        )
    else:
        for i, (exp_t, act_t) in enumerate(zip(expected["tables"], actual["tables"])):
            if exp_t["bbox"] != act_t["bbox"]:
                problems.append(
                    f"[{name}] table[{i}] bbox changed: "
                    f"expected {exp_t['bbox']}, got {act_t['bbox']}"
                )
            if exp_t["text"] != act_t["text"]:
                problems.append(
                    f"[{name}] table[{i}] text changed:\n"
                    f"  expected: {exp_t['text']!r}\n"
                    f"  actual  : {act_t['text']!r}"
                )

    if expected["blocks"] != actual["blocks"]:
        for i, (exp_b, act_b) in enumerate(
            zip(expected["blocks"], actual["blocks"])
        ):
            if exp_b != act_b:
                problems.append(
                    f"[{name}] block[{i}] changed:\n"
                    f"  expected: {exp_b}\n"
                    f"  actual  : {act_b}"
                )
        if len(expected["blocks"]) != len(actual["blocks"]):
            problems.append(
                f"[{name}] block count changed: "
                f"expected {len(expected['blocks'])}, got {len(actual['blocks'])}"
            )

    return problems


def update_baselines() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for fixture_name, pdf_name, page_index in SAMPLES:
        summary = build_summary(pdf_name, page_index)
        fixture_path = FIXTURE_DIR / f"{fixture_name}.json"
        fixture_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {fixture_path}")


def run_regression() -> None:
    print("=" * 60)
    print("DIGITAL-PDF LAYOUT REGRESSION TEST")
    print("=" * 60)

    all_problems: list = []
    for fixture_name, pdf_name, page_index in SAMPLES:
        expected = load_expected(fixture_name)
        actual = build_summary(pdf_name, page_index)
        problems = diff_summary(fixture_name, expected, actual)
        status = "OK" if not problems else "MISMATCH"
        print(
            f"{fixture_name:12s} page={page_index}  "
            f"blocks={actual['num_blocks']:3d}  tables={len(actual['tables'])}  [{status}]"
        )
        all_problems.extend(problems)

    if all_problems:
        raise AssertionError(
            f"Layout output drifted from pinned baseline in {len(SAMPLES)} sample form(s):\n\n"
            + "\n\n".join(all_problems)
        )

    print()
    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    if "--update-baseline" in sys.argv:
        update_baselines()
    else:
        run_regression()
else:
    run_regression()
