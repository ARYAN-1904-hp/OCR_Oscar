Document Router Module
Overview

The Document Router determines the appropriate OCR processing pipeline based on the uploaded document type. This abstraction allows the system to support multiple OCR engines and processing strategies without modifying the Handler.

Responsibilities
Detect document type.
Select OCR pipeline.
Update Document object.
Current Pipelines
File Type	Pipeline
PDF	PDF_OCR
Image	IMAGE_OCR
DOCX	DOC_OCR
Spreadsheet	TABLE_OCR
Text	TEXT_PROCESSOR
Workflow
Document

↓

Detect Type

↓

Select Pipeline

↓

Update Document

↓

Return
Future Enhancements
AI-based routing
Dynamic pipeline selection
Hybrid OCR routing