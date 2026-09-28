Handler Module
Overview

The Handler module serves as the primary entry point for all incoming documents in the OCR Intake Engine. It is responsible for validating uploaded files, creating document objects, managing storage, initializing metadata, routing documents to the appropriate OCR pipeline, and placing documents into the processing queue. The Handler acts as the coordinator for all preprocessing activities before OCR execution.

Responsibilities
Validate uploaded documents.
Generate unique document identifiers.
Create the Document object.
Store the original document.
Create the output directory.
Count document pages.
Route the document to the appropriate OCR pipeline.
Save document metadata.
Add the document to the processing queue.
Inputs
PDF
PNG
JPG
JPEG
BMP
TIFF
DOCX
XLSX
CSV
TXT
Outputs

A fully initialized Document object containing:

Document ID
File information
Storage path
Output directory
Page count
Pipeline information
Processing status
Metadata
Workflow
Upload Document
        │
        ▼
Validate File
        │
        ▼
Create Document Object
        │
        ▼
Store Original File
        │
        ▼
Create Output Directory
        │
        ▼
Page Counter
        │
        ▼
Document Router
        │
        ▼
Metadata Manager
        │
        ▼
Queue Manager
        │
        ▼
QUEUED
Dependencies
Config
Storage Manager
Metadata Manager
Queue Manager
Page Counter
Document Router
Future Enhancements
OCR Engine Integration
Parallel Processing
API Integration
Distributed Queue Support