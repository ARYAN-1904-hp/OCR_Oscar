"""
=========================================================
OCR Handler & Validation Manager
Project Runner
=========================================================

Run project modules and tests from the project root.

Examples:
    python run.py config
    python run.py test_config
    python run.py integration_test
    python run.py all

=========================================================
"""

import subprocess
import sys
from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================================
# Available Modules
# ==========================================================

MODULES = {
    # Core
    "config": "handler_checker.config",

    # Full clean workflow (recommended)
    "workflow": "tests.test_main",
    "workflow_fast": "tests.test_main",

    # Tests
    "test_main": "tests.test_main",
    "test_ocr_manager": "tests.test_ocr_manager",
    "test_document_router": "tests.test_document_router",
    "test_page_counter": "tests.test_page_counter",
    "test_handler": "tests.test_handler",
    "test_checker": "tests.test_checker",
    "test_retry_manager": "tests.test_retry_manager",
    "test_config": "tests.test_config",
    "test_models": "tests.test_models",
    "test_storage": "tests.test_storage",
    "test_metadata": "tests.test_metadata",
    "test_queue": "tests.test_queue",
    "integration_test": "tests.integration_test",
}

# ==========================================================
# Test Suite
# ==========================================================

ALL_TESTS = [
    "test_ocr_manager",
    "test_handler",
    "test_page_counter",
    "test_document_router",
    "test_checker",
    "test_retry_manager",
    "test_config",
    "test_models",
    "test_storage",
    "test_metadata",
    "test_queue",
    "integration_test",
    # Run full workflow last so final metadata state is COMPLETED + synced
    "test_main",
]

# ==========================================================
# Run Module
# ==========================================================

def run_module(module_name: str) -> int:
    """Run a Python module."""

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=False,
    )

    return result.returncode


# ==========================================================
# Main
# ==========================================================

def main() -> int:

    if len(sys.argv) < 2:

        print("=" * 70)
        print("OCR HANDLER & VALIDATION MANAGER")
        print("=" * 70)

        print("\nUsage:\n")

        print("  python run.py workflow        # all PDFs/images in tests/data")
        print("  python run.py workflow_fast   # PDFs only (fast)")
        print("  python run.py test_main")
        print("  python run.py all")

        print("\nAvailable Commands:\n")

        for cmd in MODULES.keys():
            print(f"  • {cmd}")

        print("  • all")

        return 1

    command = sys.argv[1].strip().lower()

    # ------------------------------------------------------

    if command == "all":

        exit_code = 0

        print("\n" + "=" * 70)
        print("RUNNING COMPLETE TEST SUITE")
        print("=" * 70)

        for test in ALL_TESTS:

            print("\n" + "=" * 70)
            print(f"Running : {test}")
            print("=" * 70)

            result = run_module(MODULES[test])

            if result != 0:
                exit_code = result

        print("\n" + "=" * 70)

        if exit_code == 0:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")

        print("=" * 70)

        return exit_code

    # ------------------------------------------------------

    if command not in MODULES:

        print(f"\nUnknown command: {command}\n")

        print("Available Commands:\n")

        for cmd in MODULES.keys():
            print(f"  • {cmd}")

        print("  • all")

        return 1

    # ------------------------------------------------------

    if command == "workflow_fast":
        import os

        os.environ["WORKFLOW_FAST"] = "1"

    return run_module(MODULES[command])


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    raise SystemExit(main())