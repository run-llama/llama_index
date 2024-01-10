import unittest
from unittest.mock import patch

from llama_index.vector_stores.pinecone import PineconeVectorStore


class MockPineconePods:
    @staticmethod
    def init(api_key: str, environment: str) -> None:
        pass

    class Index:
        def __init__(self, index_name: str) -> None:
            pass

    class Pinecone:
        def __init__(self, api_key: str) -> None:
            pass

        def Index(self, index_name: str) -> None:
            pass


class MockPineconeServerless:
    class Pinecone:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def Index(self, index_name: str) -> "MockPineconeServerless.Index":
            return MockPineconeServerless.Index(index_name)

    class Index:
        def __init__(self, index_name: str) -> None:
            self.index_name = index_name


class TestPineconeVectorStore(unittest.TestCase):
    def test_with_pod_based_true(self) -> None:
        # Testing with pod-based configuration
        with patch(
            "llama_index.vector_stores.pinecone.PineconeVectorStore._initialize_pinecone_client",
            return_value=MockPineconePods.Index("dummy_index"),
        ):
            store = PineconeVectorStore(
                api_key="dummy_key",
                index_name="dummy_index",
                environment="dummy_env",
                use_pod_based=True,
            )
            self.assertIsInstance(store._pinecone_index, MockPineconePods.Index)

    def test_with_pod_based_false(self) -> None:
        # Testing with serverless configuration
        with patch(
            "llama_index.vector_stores.pinecone.PineconeVectorStore._initialize_pinecone_client",
            return_value=MockPineconeServerless.Pinecone("dummy_key").Index(
                "dummy_index"
            ),
        ):
            store = PineconeVectorStore(
                api_key="dummy_key", index_name="dummy_index", use_pod_based=False
            )
            self.assertIsInstance(store._pinecone_index, MockPineconeServerless.Index)
