"""Shared test utilities and fixtures."""

import functools
import subprocess
from pathlib import Path

import pytest

SAMPLES_DIR = Path(__file__).parent.parent / "samples" / "pdf"


@functools.lru_cache()
def java_available() -> bool:
    """Check if Java is available on the system."""
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture()
def sample_pdf() -> Path:
    """Return path to a single-page test PDF."""
    return SAMPLES_DIR / "lorem.pdf"


@pytest.fixture()
def multi_page_pdf() -> Path:
    """Return path to a multi-page test PDF."""
    return SAMPLES_DIR / "2408.02509v1.pdf"


@pytest.fixture()
def sample_pdfs() -> list[Path]:
    """Return paths to all test PDFs."""
    return list(SAMPLES_DIR.glob("*.pdf"))
