"""
Logger configuration for the OCR Handler project.
"""

import logging

from handler_checker.config import LOG_FILE

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("handler_checker")