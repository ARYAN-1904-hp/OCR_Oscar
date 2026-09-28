Page Counter Module
Overview

The Page Counter module determines the total number of pages present in an uploaded document before OCR processing begins. Accurate page counting enables validation, processing progress tracking, and retry management.

Responsibilities
Read uploaded documents.
Determine page count.
Update the Document object.
Support multiple document formats.
Inputs

Document object

Outputs

Updated Document object containing:

original_pages
Workflow
Document

↓

Read File

↓

Count Pages

↓

Update Document

↓

Return
Dependencies
PyPDF
Document Model
Future Enhancements
Multi-format page counting
Corrupted PDF detection
Performance optimization