import os
from pathlib import Path

import pytest


def pytest_configure(config):
    """Load .env from project root so ENDEE_API_TOKEN is available."""
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env")
    except ImportError:
        pass
