"""Shared pytest configuration for the CardiaNova test suite.

The application is imported as a top-level package (`server`, `src.*`), so the
project root is put on sys.path before any test module is collected.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
