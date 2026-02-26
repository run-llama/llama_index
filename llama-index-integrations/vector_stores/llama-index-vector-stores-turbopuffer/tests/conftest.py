import os
from pathlib import Path

import pytest_asyncio  # noqa: F401

# Load .env file from the package root so TURBOPUFFER_API_KEY is available.
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.is_file():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
