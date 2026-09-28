from __future__ import annotations

import re
from typing import Dict, Any, Optional


# -------------------------
# Text cleanup
# -------------------------
_WS_RE = re.compile(r"\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

def _normalize_proc_id(m: "re.Match[str]") -> str:
    """PROC### processor/verifier id: map any O/o/0 run in the digit slots to 0."""
    digits = "".join("0" if ch in "Oo" else ch for ch in m.group(1))
    return "PROC" + digits


# Expanded OCR confusion corrections for PaddleOCR on printed English
_OCR_FIXES = [
    # Zero vs letter O
    (re.compile(r"\b0([A-Za-z])"), r"O\1"),
    (re.compile(r"\b([A-Za-z])0\b"), r"\1O"),
    # Processor/verifier id "PROC###" (e.g. Curamind claims UI): PaddleOCR
    # reads this bold all-caps token inconsistently case-by-case within the
    # SAME page - "PROc001", "PROCOO1", "PRoco01" all seen for one literal
    # "PROC001" - because the O<->0 confusion sits in the middle of one
    # contiguous alnum token, where the two \b-anchored rules above can't
    # reach (a boundary never occurs between two \w chars, and letters/
    # digits are both \w). Recognize the known "PROC" + 3 digit-like chars
    # shape directly instead of guessing at mid-token digits generically.
    (re.compile(r"\bPR[Oo0][Cc]([Oo0][Oo0-9]{2})\b"), _normalize_proc_id),
    # NOTE: a blind "l vs 1" and "rn vs m" substitution used to live here.
    # Both fire on ordinary correctly-recognized English text (e.g. "morning"
    # -> "moming", "corner" -> "comer", "1st" -> "lst", "1am" -> "lam") because
    # a regex can't distinguish a genuine OCR misread from real "rn"/leading-"1"
    # text, so they were silently corrupting far more characters than they fixed.
    # ii vs u (double lowercase-i vs u)
    (re.compile(r"\bii([a-z])"), r"u\1"),
    # Stray pipe characters from table borders
    (re.compile(r"(?<!\s)\|(?!\s)"), " "),
    # Auth/Referral misspelling (f->t OCR confusion)
    (re.compile(r'\bAuth/Reterral\b'), 'Auth/Referral'),
    # ...and the final "l" read as "!", "1", "I" or "|"
    (re.compile(r'\bAuth/Referra[!1I|]'), 'Auth/Referral'),
    # "IcD-10" / "lCD-10": ICD code-set name with a mis-cased letter
    (re.compile(r'\b[IiLl][Cc][Dd]-(9|10)\b'), r'ICD-\1'),
    # Service code line-split recovery: "P-\n99203" or "P- 99203" -> "P-99203"
    (re.compile(r'\b(P|Q|G|H|S|T|V)-\s+(\d{4,5})\b'), r'\1-\2'),
    # Leading O1/O2 in outcome codes -> 01/02 (uppercase O misread as zero context)
    (re.compile(r'\bO(\d)\s+(DISCHARGED|EXPIRED|TRANSFERRED|HOSPICE|LEFT)\b'),
     r'0\1 \2'),
    # Diagnosis code space insertion: "J0 6.9" -> "J06.9", "C1 5.9" -> "C15.9"
    (re.compile(r'\b([A-Z])(\d)\s(\d+\.\d+)\b'), r'\1\2\3'),
    # "Totals:" followed by amounts - ensure space after colon
    (re.compile(r'\bTotals:(\S)'), r'Totals: \1'),
    # EDI Claim/Batch ID spacing
    (re.compile(r'\bEDIClaim#'), 'EDI Claim # '),
    (re.compile(r'\bEDIBatchID:'), 'EDI Batch ID: '),
    # POA Indicator help text ("Y - YES, present on admission" / "N - NO, not
    # present on admission"): PaddleOCR reads the comma as a period when it
    # sits flush against the next word with no rendered gap ("YES.present").
    # This is a fixed help string in this claims UI, not free-form prose, so
    # recognizing it by name is precise where a blanket "period before a
    # lowercase letter must be a comma" rule would misfire on real
    # abbreviations elsewhere (e.g. "no.of").
    (re.compile(r"\bYES\.(present on admission)\b", re.I), r"YES, \1"),
    (re.compile(r"\bNO\.(not present on admission)\b", re.I), r"NO, \1"),
    # "appian_nold_ticket_creation" -> "appian_hold_ticket_creation" (h read as
    # n). "nold" is not an English word; lowercase only so a surname "Nold"
    # is left alone. Letter-bounded (not \b) so it also fires inside
    # snake_case identifiers.
    (re.compile(r"(?<![A-Za-z])nold(?![A-Za-z])"), "hold"),
    # "Ticket Detoils Form" -> "Ticket Details Form" (a read as o)
    (re.compile(r"\b([Dd])etoil(s?)\b"), r"\1etail\2"),
]

# -------------------------
# Claims-form structural repairs
# -------------------------
# These fix word-ORDER / word-GROUPING problems, not character misreads -
# a vertically-wrapped 2-line label ("Member" over "ID:") loses its top-to-
# bottom order once its words are re-sorted left-to-right within the merged
# row, and a form field that repeats 2-3 times side-by-side (Auth/Referral
# slots 1-3) has its per-slot labels and values interleaved the same way.
# Each pattern below targets one specific known field in this claims UI
# (matching the existing narrow-vocabulary style used elsewhere in this
# file, e.g. the Auth/Reterral and POA Indicator fixes) rather than a
# general reordering rule, which would be far too easy to misfire with.

# "ID:" landed before the label it belongs to (e.g. "ID: Member H6881954663"
# instead of "Member ID: H6881954663") because the label wrapped onto two
# lines and the second line ("ID:") sorted ahead of the first ("Member") by
# x-position once both were merged into one text row.
_ID_LABEL_REVERSAL_RE = re.compile(
    r"\bID:[ \t]+(Member|Provider|Patient|Subscriber|Policy|Group|Employee|Insured|Guarantor)\b"
)

# "Gender (3): F" read as "Gender 3) F." - the opening paren and colon are
# swapped/dropped and a stray period is appended to the value.
_GENDER_CODE_RE = re.compile(r"\bGender[ \t]+(\d+)\)[ \t]*([A-Za-z])\.?")

# Three repeated "Auth/Referral #N(23):" slots on one row get interleaved:
# "Auth/Referral #1(23): <v1> #2(23): Auth/Referral Auth/Referral #3(23): <v3>"
# Re-emit as three explicit slot lines (empty slots keep an explicit blank
# value instead of disappearing).
_AUTH_REFERRAL_JAM_RE = re.compile(
    r"Auth/Referral[ \t]+#1\(23\):[ \t]*(?P<v1>\S*)[ \t]*"
    r"#2\(23\):[ \t]*Auth/Referral[ \t]+"
    r"Auth/Referral[ \t]+#3\(23\):[ \t]*(?P<v3>\S*)"
)

# Same interleaving for the three "Auth/Ref Status:" slots that sit under
# the referral numbers above.
_AUTH_REF_STATUS_JAM_RE = re.compile(
    r"Auth/Ref Status:[ \t]*(?P<v1>\S*)[ \t]*(?P<v1b>APPROVED|DENIED|PENDING|REJECTED)?[ \t]*"
    r"Auth/Ref Status:[ \t]*"
    r"Auth/Ref Status:"
)


def _auth_referral_repl(m: "re.Match[str]") -> str:
    v1 = m.group("v1") or ""
    v3 = m.group("v3") or ""
    return (
        f"Auth/Referral #1(23): {v1}\n"
        f"Auth/Referral #2(23): \n"
        f"Auth/Referral #3(23): {v3}"
    )


def _auth_ref_status_repl(m: "re.Match[str]") -> str:
    v1 = (m.group("v1") or "").strip()
    v1b = (m.group("v1b") or "").strip()
    val = (v1 + " " + v1b).strip()
    return (
        f"Auth/Ref Status: {val}\n"
        f"Auth/Ref Status: \n"
        f"Auth/Ref Status: "
    )


# A short code immediately followed by its all-caps description reads as
# one run of text with no marked boundary (e.g. "Status: 2 PENDING REVIEW").
# Insert an explicit separator so the code and description don't have to be
# told apart by whitespace-only heuristics downstream. Scoped to the known
# code/description fields in this claims UI; the "Status:" pattern excludes
# "Auth/Ref Status:" (handled by its own jam-fix above) via lookbehind.
_CODE_DESC_BOUNDARY_FIXES = [
    (re.compile(r"\bCompany ID:[ \t]*([A-Z0-9]{2,10})[ \t]+(?=[A-Z][A-Z0-9 ]{2,30}\b)"), r"Company ID: \1 | "),
    (re.compile(r"(?<!Auth/Ref )\bStatus:[ \t]*(\d{1,3})[ \t]+(?=[A-Z][A-Z ]{2,30}\b)"), r"Status: \1 | "),
    (re.compile(r"\bPriority:[ \t]*(\d{1,3})[ \t]+(?=[A-Z][A-Z ]{2,30}\b)"), r"Priority: \1 | "),
]


def fix_claims_form_structure(text: str) -> str:
    """Repair word-order/word-grouping issues specific to this claims UI.

    Runs after fix_common_paddle_errors() (character-level fixes); these
    patterns depend on field names being already spelled correctly.
    """
    if not text:
        return text
    t = text
    t = _ID_LABEL_REVERSAL_RE.sub(lambda m: f"{m.group(1)} ID:", t)
    t = _GENDER_CODE_RE.sub(lambda m: f"Gender ({m.group(1)}): {m.group(2)}", t)
    t = _AUTH_REFERRAL_JAM_RE.sub(_auth_referral_repl, t)
    t = _AUTH_REF_STATUS_JAM_RE.sub(_auth_ref_status_repl, t)
    for pat, repl in _CODE_DESC_BOUNDARY_FIXES:
        t = pat.sub(repl, t)
    return t


_PADDLE_SPACE_FIXES = [
    # Fix missing space before capital in middle of word: "helloWorld" -> "hello World"
    (re.compile(r"([a-z])([A-Z][a-z])"), r"\1 \2"),
    # Two sentences read as one token with the gap between them lost
    # ("Documentation complete.Approved..."): a letter/digit immediately
    # followed by sentence-ending punctuation and then a capitalized word is,
    # in printed body text, always a missing space rather than a real word -
    # safe to insert unconditionally (unlike the comma/period confusion
    # above, this doesn't touch what character was recognized, only
    # restores the space PaddleOCR dropped between two adjacent OCR lines).
    (re.compile(r"([a-zA-Z0-9])([.!?])([A-Z][a-z])"), r"\1\2 \3"),
    # Safety nets for spaces the recognizer drops when a box is cropped
    # tightly (ocr/recognition_refine.py restores most of these from the
    # image; these cover what it can't re-read with confidence).
    # "11/17/202509:12" -> "11/17/2025 09:12", "7/18/20267:02PM" -> "7/18/2026 7:02PM"
    # (no trailing \b: the time is often glued to AM/PM, handled below)
    (re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})(\d{1,2}:\d{2})(?!\d)"), r"\1 \2"),
    # "2026-07-23.20:15:51" -> "2026-07-23 20:15:51" (space misread as "."/",")
    (re.compile(r"\b(\d{4}-\d{2}-\d{2})[.,](\d{1,2}:\d{2})"), r"\1 \2"),
    # "7:02PMEDT" -> "7:02 PM EDT", "09:12AM" -> "09:12 AM"
    (re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)[ \t]*([AP]M)[ \t]*([ECMP][SD]T|UTC|GMT)?\b"),
     lambda m: " ".join(g for g in m.groups() if g)),
    # "HOLD-5886366Production Hold" -> "HOLD-5886366 Production Hold": an ID's
    # digits glued to a following capitalized word
    (re.compile(r"(?<=\d)(?=[A-Z][a-z]{2,})"), " "),
    # "Queue:18", "Status:M", "Verified By:PROC002" -> "Label: value".
    # Only after a letter/")" (so times like 09:12 are untouched) and never
    # inside a URL scheme.
    (re.compile(r"(?<=[A-Za-z)])(?<!http)(?<!https)(?<!ftp)(?<!mailto):(?=[A-Za-z0-9$(])"), ": "),
    # "DAVIS,OLIVER", "Sacramento,CA" -> comma-space between words
    # (digits excluded so amounts like 1,840.96 are untouched)
    (re.compile(r"(?<=[A-Za-z]),(?=[A-Za-z])"), ", "),
    # "SACRAMENTO, CA95182" -> "SACRAMENTO, CA 95182" (state + ZIP after a comma)
    (re.compile(r",\s*([A-Z]{2})(\d{5}(?:-\d{4})?)\b"), r", \1 \2"),
    # "OBSTETRICS &GYNECOLOGY" -> "OBSTETRICS & GYNECOLOGY"
    (re.compile(r"(?<=\s)&(?=[A-Za-z])"), "& "),
    # "Member/ Patient Information" -> "Member / Patient Information"
    (re.compile(r"(?<=\w)/ (?=\w)"), " / "),
    # "$0.00/$750.00" -> "$0.00 / $750.00" (amount-of-total pairs)
    (re.compile(r"(?<=\d)\s*/\s*(?=\$)"), " / "),
    # "(PROC002) |Email:" -> "(PROC002) | Email:" (UI field separator)
    (re.compile(r"(?<=\s)\|(?=[A-Za-z])"), "| "),
    # "ProcessorID", "PatientID" -> "Processor ID" (acronym glued to a word)
    (re.compile(r"\b([A-Z][a-z]{2,})(ID|NPI|SSN|DOB)\b"), r"\1 \2"),
    # "ClaimsLM" -> "Claims LM" (menu label glued to a 2-3 letter avatar /
    # acronym). Needs 2+ lowercase letters first so "PhD" is untouched.
    (re.compile(r"\b([A-Z][a-z]{2,})([A-Z]{2,3})\b"), r"\1 \2"),
    # "Resolved(Suppressed)" -> "Resolved (Suppressed)". Capitalized word
    # inside the parens only, so "claim(s)" and "#2(23)" are untouched.
    (re.compile(r"(?<=[a-z])\((?=[A-Z][a-z])"), " ("),
    # "(minus High$)(7 working days)" -> "... High$) (7 working days)"
    (re.compile(r"\)\((?=[A-Za-z0-9])"), ") ("),
    # "ofthe claim" -> "of the claim": preposition fused with "the". None of
    # these fused forms is a real word.
    (re.compile(r"\b(of|in|on|to|for|at|by|from|with|and|is)the\b", re.I), r"\1 the"),
    # "for anewborn" -> "for a newborn": article fused with a claims-domain
    # noun. A fixed list, because a general "a<word>" split would break
    # "another", "about", "anew", ...
    (re.compile(r"\b([Aa])(newborn|claim|policy|member|patient|provider|dependent|baby|child)\b"),
     r"\1 \2"),
]

# Phone/fax numbers whose area-code parentheses were dropped or half-read by
# the recognizer ("555881-9263", "555)593-7414"). Scoped to a Phone/Fax/Tel
# label on the same line so arbitrary 6+4 digit IDs are never reformatted.
_PHONE_NO_PARENS_RE = re.compile(
    r"(?P<label>\b(?:Phone|Fax|Tel|Telephone|Mobile|Cell|Ph)\b[ \t#.:]*(?:No\.?|Number)?[ \t:]*)"
    r"\(?(?P<area>\d{3})\)?(?P<ex>\d{3})-(?P<num>\d{4})\b",
    re.I,
)


def _fix_phone_parens(text: str) -> str:
    return _PHONE_NO_PARENS_RE.sub(
        lambda m: f"{m.group('label')}({m.group('area')}) {m.group('ex')}-{m.group('num')}", text
    )


def clean_text(text: str) -> str:
    """Lightweight cleanup without being too destructive."""
    if not text:
        return ""
    t = text.replace("\u00a0", " ")  # NBSP
    t = _NON_ASCII_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


def fix_common_paddle_errors(text: str) -> str:
    """Apply PaddleOCR-specific post-correction heuristics."""
    if not text:
        return ""
    t = text
    for pat, repl in _OCR_FIXES:
        t = pat.sub(repl, t)
    for pat, repl in _PADDLE_SPACE_FIXES:
        t = pat.sub(repl, t)
    t = _fix_phone_parens(t)
    return t


def normalize_linebreaks(text: str) -> str:
    """Keep paragraphs but normalize excessive blank lines, then apply paddle fixes."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = fix_common_paddle_errors(t)
    t = fix_claims_form_structure(t)
    return t.strip()


# -------------------------
# Field validation / normalization
# -------------------------
PIN_RE = re.compile(r"\b\d{6}\b")                           # India PIN
ALNUM_ID_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9\-\/]{5,}\b")  # IDs like ABC-123/45
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)     # PAN
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)   # IFSC
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?(?:0[\s-]?)?\b\d{10}\b")


def _first_match(pattern: re.Pattern, value: str) -> Optional[str]:
    if not value:
        return None
    m = pattern.search(value)
    return m.group(0) if m else None


def _apply_ocr_fixes(value: str) -> str:
    v = (value or "").strip()
    for pat, repl in _OCR_FIXES:
        v = pat.sub(repl, v)
    return v


def validate_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort validators. Never delete fields; only normalize when confident."""
    if not fields:
        return fields

    out: Dict[str, Any] = dict(fields)

    # Generic normalization
    for k, v in list(out.items()):
        if isinstance(v, str):
            out[k] = clean_text(v)

    # Pincode
    for key in ["Pincode", "Pin", "PIN", "Pin Code", "PIN Code"]:
        if key in out and isinstance(out[key], str):
            m = _first_match(PIN_RE, out[key])
            if m:
                out["Pincode"] = m

    # Policy / IDs (generic)
    for key in ["Policy No", "Policy Number", "Certificate No", "SL No/Certificate No", "Company/TPA ID No", "TPA ID"]:
        if key in out and isinstance(out[key], str):
            v = _apply_ocr_fixes(out[key]).replace(" ", "")
            m = _first_match(ALNUM_ID_RE, v)
            if m:
                out[key] = m

    # PAN
    if "PAN" in out and isinstance(out["PAN"], str):
        m = _first_match(PAN_RE, out["PAN"].upper().replace(" ", ""))
        if m:
            out["PAN"] = m.upper()

    # IFSC
    if "IFSC" in out and isinstance(out["IFSC"], str):
        m = _first_match(IFSC_RE, out["IFSC"].upper().replace(" ", ""))
        if m:
            out["IFSC"] = m.upper()

    # Email
    if "Email" in out and isinstance(out["Email"], str):
        m = _first_match(EMAIL_RE, out["Email"])
        if m:
            out["Email"] = m

    # Phone/Mobile
    for key in ["Mobile", "Phone", "Contact", "Mobile No", "Phone No"]:
        if key in out and isinstance(out[key], str):
            m = _first_match(PHONE_RE, out[key].replace(" ", ""))
            if m:
                out[key] = re.sub(r"\D", "", m)[-10:]

    return out
