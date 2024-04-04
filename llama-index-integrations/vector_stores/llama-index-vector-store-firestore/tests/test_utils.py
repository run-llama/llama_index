from unittest import TestCase
from unittest.mock import patch
import pytest
from llama_index.vector_store.firestore.utils import client_with_user_agent


@pytest.fixture(autouse=True, name="test_case")
def init_test_case() -> TestCase:
    """Returns a TestCase instance."""
    return TestCase()


def test_user_agent_with_correct_version(test_case: TestCase) -> None:
    """Test that the user agent includes the correct version."""
    with patch("importlib.metadata.version", return_value="0.1.0"):
        client = client_with_user_agent()
        test_case.assertEqual(
            client._client_info.user_agent,
            "llama-index-vector-store-firestore-python:vectorstore0.1.0",
        )
