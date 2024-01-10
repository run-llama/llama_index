import os
import unittest
from typing import Optional
from unittest.mock import patch

import pytest
from llama_index.vector_stores.pinecone import PineconeVectorStore


class MockPineconePods:
    @staticmethod
    def init(api_key: Optional[str], environment: str) -> None:
        pass

    class Index:
        def __init__(self, index_name: str) -> None:
            pass

    class Pinecone:
        def __init__(self, api_key: Optional[str]) -> None:
            pass

        def Index(self, index_name: str) -> None:
            pass


class MockPineconeServerless:
    class Pinecone:
        def __init__(self, api_key: Optional[str]) -> None:
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
        # Testing with serverless configuration (False is default, so no need to pass param)
        with patch(
            "llama_index.vector_stores.pinecone.PineconeVectorStore._initialize_pinecone_client",
            return_value=MockPineconeServerless.Pinecone("dummy_key").Index(
                "dummy_index"
            ),
        ):
            store = PineconeVectorStore(api_key="dummy_key", index_name="dummy_index")
            self.assertIsInstance(store._pinecone_index, MockPineconeServerless.Index)

    def test_str_value_for_pinecone_index_param_does_not_match_str_value_for_index_name_param(
        self,
    ) -> None:
        os.environ["PINECONE_API_KEY"] = "some-key"
        api_key = os.getenv("PINECONE_API_KEY")
        str_pinecone_index = "some-string-representing-an-existing-pinecone-index"
        with patch(
            "llama_index.vector_stores.pinecone.PineconeVectorStore._initialize_pinecone_client",
            return_value=MockPineconeServerless.Pinecone(api_key).Index("dummy_index"),
        ):  # MockPineconeServerless vs MockPineconePods is arbitrary for this test
            with pytest.raises(ValueError) as e:
                store = PineconeVectorStore(
                    pinecone_index=str_pinecone_index,
                    api_key=api_key,
                    index_name="dummy_index",
                    use_pod_based=False,
                )
                assert (
                    e.value.message
                    == "The string value for `pinecone_index` must match the string value for `index_name`."
                )

    def test_str_value_for_pinecone_index_param_gets_transformed_into_index_obj(
        self,
    ) -> None:
        os.environ["PINECONE_API_KEY"] = "some-key"
        api_key = os.getenv("PINECONE_API_KEY")
        str_pinecone_index = "some-string-representing-an-existing-pinecone-index"
        with patch(
            "llama_index.vector_stores.pinecone.PineconeVectorStore._initialize_pinecone_client",
            return_value=MockPineconeServerless.Pinecone(api_key).Index(
                str_pinecone_index
            ),
        ):  # MockPineconeServerless vs MockPineconePods is arbitrary for this test
            store = PineconeVectorStore(
                pinecone_index=str_pinecone_index,
                api_key=api_key,
                index_name=str_pinecone_index,
            )
            self.assertIsInstance(store._pinecone_index, MockPineconeServerless.Index)

    def test_pinecone_vector_store_without_passing_index_name(self) -> None:
        with pytest.raises(ValueError) as e:
            PineconeVectorStore()
            assert (
                e.value.message
                == "index_name is required for Pinecone client initialization"
            )
