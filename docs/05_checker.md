Checker Module
Overview

The Checker module validates the results of OCR processing before marking a document as successfully processed. It ensures document completeness and verifies that processing outputs satisfy predefined validation rules.

Responsibilities
Validate page count.
Validate output directory.
Verify OCR completion.
Update validation status.
Record validation errors.
Validation Rules

Current implementation:

Output directory exists.
Original pages equal processed pages.
Outputs

Updates:

validation_passed
status
error_message
Workflow
OCR Output

↓

Page Validation

↓

Output Validation

↓

PASS / FAIL

↓

Update Document
Future Enhancements
JSON validation
Confidence threshold
Entity validation
Business rule validation