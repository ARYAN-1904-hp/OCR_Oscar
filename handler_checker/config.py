"""
=========================================================
OCR Handler + Validation Manager
Configuration Manager
=========================================================

Loads all configurable values from the .env file.

Responsibilities
----------------
• Load environment variables
• Create required project directories
• Initialize metadata, queue and log files
• Expose configuration constants to the application
=========================================================
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# ==========================================================
# Load Environment Variables
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# ==========================================================
# Project
# ==========================================================

PROJECT_NAME = os.getenv("PROJECT_NAME", "Intake Agent OCR Engine")

# ==========================================================
# Storage
# ==========================================================

STORAGE_DIR = PROJECT_ROOT / os.getenv("STORAGE_DIR", "storage")
ORIGINALS_DIR = PROJECT_ROOT / os.getenv("ORIGINALS_DIR", "storage/originals")
OUTPUTS_DIR = PROJECT_ROOT / os.getenv("OUTPUTS_DIR", "storage/outputs")
# One folder (+ zip) per upload request, one cleaned <filename>.txt per document
TEXT_COLLECTIONS_DIR = PROJECT_ROOT / os.getenv("TEXT_COLLECTIONS_DIR", "storage/text_collections")

# ==========================================================
# Metadata
# ==========================================================

METADATA_DIR = PROJECT_ROOT / os.getenv("METADATA_DIR", "storage/metadata")
METADATA_FILE = PROJECT_ROOT / os.getenv("METADATA_FILE", "storage/metadata/metadata.json")

# ==========================================================
# Queue
# ==========================================================

QUEUE_DIR = PROJECT_ROOT / os.getenv("QUEUE_DIR", "storage/queue")
QUEUE_FILE = PROJECT_ROOT / os.getenv("QUEUE_FILE", "storage/queue/queue.json")
QUEUE_MODE = os.getenv("QUEUE_MODE", "auto")

# ==========================================================
# Logs
# ==========================================================

LOG_DIR = PROJECT_ROOT / os.getenv("LOG_DIR", "storage/logs")
LOG_FILE = PROJECT_ROOT / os.getenv("LOG_FILE", "storage/logs/app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ==========================================================
# Retry
# ==========================================================

MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "3"))

# ==========================================================
# OCR
# ==========================================================

OCR_ENGINE = os.getenv("OCR_ENGINE", "paddleocr")
OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "120"))

# ==========================================================
# Supported Extensions
# ==========================================================

SUPPORTED_EXTENSIONS = tuple(
    ext.strip().lower()
    for ext in os.getenv(
        "SUPPORTED_EXTENSIONS", ".pdf,.png,.jpg,.jpeg,.docx,.pptx,.doc,.ppt,.xlsx"
    ).split(",")
)

# ==========================================================
# Create Required Directories
# ==========================================================

for directory in [
    STORAGE_DIR,
    ORIGINALS_DIR,
    OUTPUTS_DIR,
    TEXT_COLLECTIONS_DIR,
    METADATA_DIR,
    QUEUE_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Initialize Files
# ==========================================================

if not METADATA_FILE.exists():
    METADATA_FILE.write_text("{}", encoding="utf-8")

if not QUEUE_FILE.exists():
    QUEUE_FILE.write_text("[]", encoding="utf-8")

if not LOG_FILE.exists():
    LOG_FILE.touch()


if __name__ == "__main__":
    print("Project Name :", PROJECT_NAME)
    print("Project Root :", PROJECT_ROOT)
    print("Storage      :", STORAGE_DIR)
    print("Metadata     :", METADATA_FILE)
    print("Queue        :", QUEUE_FILE)
    print("Log          :", LOG_FILE)
    print("Supported    :", SUPPORTED_EXTENSIONS)
    print("Retry Count  :", MAX_RETRY_COUNT)
    print("Queue Mode   :", QUEUE_MODE)
    print("OCR Engine   :", OCR_ENGINE)
    print("OCR Timeout  :", OCR_TIMEOUT)