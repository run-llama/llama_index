from typing import Any
from unittest import TestCase
from unittest.mock import patch
from google.cloud.firestore import Client

import pytest

from llama_index.vector_stores.firestore.utils import USER_AGENT


@pytest.fixture(autouse=True, name="test_case")
def init_test_case() -> TestCase:
    """Returns a TestCase instance."""
    return TestCase()


@pytest.fixture(autouse=True, name="mock_client")
def mock_firestore_client() -> Any:
    """Returns a mock Firestore client."""
    with patch("google.cloud.firestore.Client") as mock_client_cls:
        with patch("importlib.metadata.version", return_value="0.1.0", autospec=True):
            mock_client = mock_client_cls.return_value
            mock_client._client_info.user_agent = USER_AGENT
            yield mock_client


def test_user_agent_with_correct_version(
    test_case: TestCase, mock_client: Client
) -> None:
    """Test that the user agent includes the correct version."""
    test_case.assertIn(
        "llama-index-vector-store-firestore-python:vectorstore",
        mock_client._client_info.user_agent,
    )
