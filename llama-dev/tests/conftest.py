import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def data_path():
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_current_version(monkeypatch):
    """Fixture to mock the current Python version."""

    def _set_version(major, minor, micro):
        mock_version = type(
            "MockVersion", (), {"major": major, "minor": minor, "micro": micro}
        )
        monkeypatch.setattr(sys, "version_info", mock_version)

    return _set_version


@pytest.fixture
def mock_rich_console():
    """Fixture to mock the rich console."""
    return MagicMock()
