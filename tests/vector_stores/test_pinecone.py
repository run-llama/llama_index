import builtins
import unittest
from typing import Any, Callable
from unittest.mock import patch

from llama_index.vector_stores.pinecone import (
    PineconeVectorStore,  # Replace with the actual import
)


class MockPineconePods:
    version = "2.9.9"

    @staticmethod
    def init(api_key: str, environment: str) -> None:
        pass

    class Index:
        def __init__(self, index_name: str) -> None:
            pass


class MockPineconeServerless:
    version = "3.0.0"

    class Pinecone:
        def __init__(self, api_key: str) -> None:
            pass

        class Index:
            def __init__(self, index_name: str) -> None:
                pass


# Define the mock import function
def mock_import(name: str, *args: Any, **kwargs: Any) -> Callable:
    if name == "pinecone":
        return MockPineconePods if pods_version else MockPineconeServerless  # type: ignore[name-defined]
    return original_import(name, *args, **kwargs)  # type: ignore[name-defined]


class TestPineconeVectorStore(unittest.TestCase):
    def setUp(self) -> None:
        global original_import
        original_import = builtins.__import__  # type: ignore[name-defined]

    def tearDown(self) -> None:
        builtins.__import__ = original_import  # type: ignore[name-defined]

    def test_pods_version(self) -> None:
        global pods_version
        pods_version = True  # type: ignore[name-defined]
        with patch("builtins.__import__", side_effect=mock_import):
            store = PineconeVectorStore(
                api_key="dummy_key", index_name="dummy_index", environment="dummy_env"
            )

    def test_serverless_version(self) -> None:
        global pods_version
        pods_version = False  # type: ignore[name-defined]
        with patch("builtins.__import__", side_effect=mock_import):
            store = PineconeVectorStore(api_key="dummy_key", index_name="dummy_index")
