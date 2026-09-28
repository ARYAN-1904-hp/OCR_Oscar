Retry Manager Module
Overview

The Retry Manager manages document reprocessing when validation fails. It prevents infinite retries while providing automatic recovery for transient processing failures.

Responsibilities
Monitor validation results.
Maintain retry count.
Retry failed processing.
Enforce retry limits.
Update document status.
Retry Rules

Maximum retries:

3

Logic:

Validation Passed

↓

Completed

Validation Failed

↓

Retry Count < Maximum

↓

Retry

Retry Count == Maximum

↓

Failed
Outputs

Updates:

retry_count
status
Future Enhancements
Page-level retry
OCR worker retry
Exponential backoff
Retry scheduling