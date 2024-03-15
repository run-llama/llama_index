import builtins
import unittest
from typing import Any, Callable, Type
from unittest.mock import patch

import pytest
from llama_index.legacy.vector_stores.pinecone import (
    PineconeVectorStore,
)


class MockPineconePods:
    __version__ = "2.2.4"

    @staticmethod
    def init(api_key: str, environment: str) -> None:
        pass

    class Index:
        def __init__(self, index_name: str) -> None:
            pass


class MockPineconeServerless:
    __version__ = "3.0.0"

    class Pinecone:
        def __init__(self, api_key: str) -> None:
            pass

    class Index:
        def __init__(self, index_name: str) -> None:
            pass


class MockUnVersionedPineconeRelease:
    @staticmethod
    def init(api_key: str, environment: str) -> None:
        pass

    class Index:
        def __init__(self, index_name: str) -> None:
            pass


def get_version_attr_from_mock_classes(mock_class: Type[Any]) -> str:
    if not hasattr(mock_class, "__version__"):
        raise AttributeError(
            "The version of pinecone you are using does not contain necessary __version__ attribute."
        )
    return mock_class.__version__


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
            mocked_version = get_version_attr_from_mock_classes(MockPineconePods)

            assert mocked_version == "2.2.4"

            # PineconeVectorStore calls its own init method when instantiated
            store = PineconeVectorStore(
                api_key="dummy_key",
                index_name="dummy_index",
                environment="dummy_env",
                pinecone_index=MockPineconePods.Index("some-pinecone-index"),
            )

    def test_serverless_version(self) -> None:
        global pods_version
        pods_version = False  # type: ignore[name-defined]
        with patch("builtins.__import__", side_effect=mock_import):
            mock_version = get_version_attr_from_mock_classes(MockPineconeServerless)

            assert mock_version == "3.0.0"

            store = PineconeVectorStore(
                api_key="dummy_key",
                index_name="dummy_index",
                pinecone_index=MockPineconeServerless.Index("some-pinecone-index"),
            )

    def test_unversioned_pinecone_client(self) -> None:
        with pytest.raises(
            AttributeError,
            match="The version of pinecone you are using does not contain necessary __version__ attribute.",
        ):
            get_version_attr_from_mock_classes(MockUnVersionedPineconeRelease)
