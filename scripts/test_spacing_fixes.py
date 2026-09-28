"""Regression checks for the spacing / punctuation / stacked-label fixes.

Run from the project root:  python scripts/test_spacing_fixes.py
No OCR models needed - the cases are real reads captured from the Curamind
claims screenshots (frame_0001..0005).
"""
import sys

sys.path.insert(0, ".")

from ocr.layout import merge_stacked_label_tokens
from ocr.postprocess import normalize_linebreaks
from ocr.recognition_refine import _edit_is_safe, _needs_reread

failures = []

# (first read, padded re-read, first conf, re-read conf)
ACCEPT = [
    ("555881-9263", "(555) 881-9263", .99, .94),
    ("PATELMD,RAJ KUMAR", "PATEL MD, RAJ KUMAR", .93, .97),
    ("PARTIALAPPROVAL", "PARTIAL APPROVAL", 1.0, .96),
    ("$2.844.42", "$2,844.42", .94, .98),
    ("#2(23)", "#2(23):", .85, .99),
    ("P3CA", "P3 CA", .99, .87),
    ("ACUTE UPPER RESPIRAT ORY INFECTION,UNSPECIFIED", "ACUTE UPPER RESPIRATORY INFECTION, UNSPECIFIED", .96, .99),
]
REJECT = [
    ("PENDING REVIEW", "PEnDing Review", .98, .66),        # case changed
    ("MARTINEZ MD,ANNA LUCIA", "MARTINEZ MD, ANNA LUCIAE", .95, .97),  # glyph added
    ("$1,840.96", "$1.840.96", .96, .96),                  # swap without better conf
    ("555593-7414", "555)593-7414", .99, .97),             # half a bracket pair
    ("Term", "Term.", .97, .96),                           # border read as "."
    ("Professional", "Professio nal", 1.0, .94),           # word split
    ("Medical Necessity Review", "Medical NecessityReview", .96, .99),
]
for old, new, c0, c1 in ACCEPT:
    if not _edit_is_safe(old, new, c0, c1):
        failures.append(f"should accept {old!r} -> {new!r}")
for old, new, c0, c1 in REJECT:
    if _edit_is_safe(old, new, c0, c1):
        failures.append(f"should reject {old!r} -> {new!r}")

for t in ["555881-9263", "PARTIALAPPROVAL", "PATELMD,RAJ KUMAR", "$2.844.42", "P3CA", "#2(23)"]:
    if not _needs_reread(t):
        failures.append(f"should re-read {t!r}")
for t in ["WILLIAMS, SAMUEL", "Professional", "11/07/2025", "PENDING REVIEW", "$1,840.96"]:
    if _needs_reread(t):
        failures.append(f"should not re-read {t!r}")

TEXT = {
    "Phone: 555881-9263": "Phone: (555) 881-9263",
    "Fax: 555)593-7414": "Fax: (555) 593-7414",
    "Last edit:11/17/202509:12": "Last edit: 11/17/2025 09:12",
    "ProcessorID:PROc002": "Processor ID: PROC002",
    "Queue:18 claims at 09:12 https://x.com": "Queue: 18 claims at 09:12 https://x.com",
    "DAVIS,OLIVER $1,840.96": "DAVIS, OLIVER $1,840.96",
    "SACRAMENTO,CA95182": "SACRAMENTO, CA 95182",
    "OBSTETRICS &GYNECOLOGY AT&T": "OBSTETRICS & GYNECOLOGY AT&T",
    "Deductible Met: $0.00/$750.00": "Deductible Met: $0.00 / $750.00",
    "Diagnosis Codes (IcD-10)": "Diagnosis Codes (ICD-10)",
    "Auth/Referra! #1(23):": "Auth/Referral #1(23):",
    "HOLD-5886366Production Hold": "HOLD-5886366 Production Hold",
    "Created On 7/18/20267:02PMEDT": "Created On 7/18/2026 7:02 PM EDT",
    "Resolved On 7/23/20261:27PMEDT": "Resolved On 7/23/2026 1:27 PM EDT",
    "2026-07-23.20:15:51 s": "2026-07-23 20:15:51 s",
    "Last edit: 11/17/2025 09:12AM": "Last edit: 11/17/2025 09:12 AM",
    "Claim Q20K5GSF P3 CA95182": "Claim Q20K5GSF P3 CA95182",
    "Start DOS ofthe claim": "Start DOS of the claim",
    "The claim is for anewborn": "The claim is for a newborn",
    "another about anew other": "another about anew other",
    "Automat (appian_nold_ticket_creation)": "Automat (appian_hold_ticket_creation)",
    "Dr. Nold": "Dr. Nold",
    "Ticket Detoils Form": "Ticket Details Form",
    "P3 Resolved(Suppressed)": "P3 Resolved (Suppressed)",
    "(minus High$)(7 working days)": "(minus High$) (7 working days)",
    "claim(s) #2(23)": "claim(s) #2(23)",
    "Claims ClaimsLM appian": "Claims Claims LM appian",
    "PhD PATIENT": "PhD PATIENT",
}
for src, want in TEXT.items():
    got = normalize_linebreaks(src)
    if got != want:
        failures.append(f"text {src!r}: got {got!r}, want {want!r}")


def w(text, x1, y1, x2, y2):
    return {"text": text, "conf": 0.99, "bbox": [x1, y1, x2, y2]}


# Left-aligned label column: each wrapped pair must merge top-then-bottom even
# though the whole column is x-aligned like a table column.
labels = [
    w("Member", 40, 280, 94, 297), w("ID:", 40, 297, 60, 312),
    w("Date of", 40, 326, 90, 342), w("Birth:", 40, 342, 79, 358),
    w("Plan", 39, 371, 70, 388), w("Type:", 38, 387, 79, 407),
    w("Group", 38, 415, 81, 437), w("#:", 40, 435, 59, 450),
    w("City/State/", 489, 550, 557, 566), w("ZIP:", 487, 565, 517, 583),
    w("Network", 450, 476, 509, 495), w("Tier:", 448, 490, 484, 512),
]
got = sorted(t["text"] for t in merge_stacked_label_tokens(labels))
want = sorted(["Member ID:", "Date of Birth:", "Plan Type:", "Group #:", "City/State/ZIP:", "Network Tier:"])
if got != want:
    failures.append(f"stacked labels: got {got}")

# A table header over its single data row must NOT merge ("Seq 1", ...), and
# two stacked address values in separate form rows must stay separate.
table = [
    w("Seq", 49, 626, 84, 649), w("CPT/HCPCS", 108, 629, 190, 645), w("Modifier", 253, 629, 315, 645),
    w("1", 50, 654, 64, 671), w("P-99214", 107, 654, 163, 670), w("59", 252, 652, 274, 672),
    w("457 Pine St, Sacramento,CA", 172, 378, 371, 397), w("457 Pine St, Sacramento,CA", 170, 408, 373, 428),
]
if len(merge_stacked_label_tokens(table)) != len(table):
    failures.append("table header / stacked values were merged")

if failures:
    print("FAILED:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("all spacing-fix checks passed")
