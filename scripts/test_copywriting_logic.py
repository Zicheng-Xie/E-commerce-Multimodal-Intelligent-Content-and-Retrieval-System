"""Compatibility command for the test suite now located in tests/unit."""

import unittest
from pathlib import Path


def main() -> None:
    """Discover and run copywriting tests, preserving the exit status."""
    test_directory = Path(__file__).resolve().parents[1] / "tests" / "unit"
    suite = unittest.defaultTestLoader.discover(
        str(test_directory), pattern="test_copywriting_logic.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
