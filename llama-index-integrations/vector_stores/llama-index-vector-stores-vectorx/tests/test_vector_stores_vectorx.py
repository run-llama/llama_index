import unittest
import sys
import pytest
import time
import os
from pathlib import Path
from unittest.mock import MagicMock

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from vecx.vectorx import VectorX
from llama_index.vector_stores.vectorx import VectorXVectorStore
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    VectorStoreQuery,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# ---- Check if credentials exist ----
HAS_VECX = os.getenv("VECTORX_API_TOKEN") is not None


# ------------------ Base Test Setup ------------------
@pytest.mark.skipif(not HAS_VECX, reason="VECTORX_API_TOKEN not set in environment")
class VectorXTestSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vecx_api_token = os.getenv("VECTORX_API_TOKEN")
        if not cls.vecx_api_token:
            raise ValueError(
                "Missing VECTORX_API_TOKEN. Please set it in your environment."
            )

        cls.vx = VectorX(token=cls.vecx_api_token)
        cls.encryption_key = cls.vx.generate_key()

        timestamp = int(time.time())
        cls.test_index_name = f"test_index_{timestamp}"
        cls.dimension = 384
        cls.space_type = "cosine"

        cls.test_indexes = {cls.test_index_name}

        cls.test_documents = [
            Document(
                text="Python is a high-level, interpreted programming language known for its readability and simplicity.",
                metadata={
                    "category": "programming",
                    "language": "python",
                    "difficulty": "beginner",
                },
            ),
            Document(
                text="Machine learning algorithms learn patterns from data to make predictions.",
                metadata={
                    "category": "ai",
                    "field": "machine_learning",
                    "difficulty": "intermediate",
                },
            ),
            Document(
                text="Deep learning uses neural networks with multiple layers for complex pattern recognition.",
                metadata={
                    "category": "ai",
                    "field": "deep_learning",
                    "difficulty": "advanced",
                },
            ),
        ]

    @classmethod
    def tearDownClass(cls):
        for index_name in cls.test_indexes:
            try:
                cls.vx.delete_index(name=index_name)
            except Exception as e:
                if "not found" not in str(e).lower():
                    print(f"Error deleting test index {index_name}: {e}")

    def tearDown(self):
        try:
            indexes = self.vx.list_indexes()
            if isinstance(indexes, list):
                for index in indexes:
                    if isinstance(index, dict) and "name" in index:
                        index_name = index["name"]
                        if index_name.startswith("test_index_"):
                            try:
                                self.vx.delete_index(name=index_name)
                            except Exception as e:
                                print(f"Error cleaning up test index {index_name}: {e}")
        except Exception as e:
            print(f"Error listing indexes for cleanup: {e}")


# ------------------ VectorX VectorStore Tests ------------------
class TestVectorXVectorStore(VectorXTestSetup):
    def setUp(self):
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )

    def test_create_vector_store_from_params(self):
        vector_store = VectorXVectorStore.from_params(
            api_token=self.vecx_api_token,
            index_name=self.test_index_name,
            encryption_key=self.encryption_key,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        self.assertIsNotNone(vector_store)
        self.assertEqual(vector_store.index_name, self.test_index_name)

    def test_create_vector_store_with_documents(self):
        vector_store = VectorXVectorStore.from_params(
            api_token=self.vecx_api_token,
            index_name=self.test_index_name,
            encryption_key=self.encryption_key,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            self.test_documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )
        self.assertIsNotNone(index)

    def test_invalid_params(self):
        with pytest.raises(Exception):
            VectorXVectorStore.from_params(
                api_token="invalid:invalid:region",
                index_name=self.test_index_name,
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type=self.space_type,
            )


# ------------------ Custom Retrieval Tests ------------------
class TestCustomRetrieval(VectorXTestSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.embed_model = HuggingFaceEmbedding(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        cls.vector_store = VectorXVectorStore.from_params(
            api_token=cls.vecx_api_token,
            index_name=cls.test_index_name,
            encryption_key=cls.encryption_key,
            dimension=cls.dimension,
            space_type=cls.space_type,
        )
        cls.storage_context = StorageContext.from_defaults(
            vector_store=cls.vector_store
        )
        Settings.llm = None
        cls.index = VectorStoreIndex.from_documents(
            cls.test_documents,
            storage_context=cls.storage_context,
            embed_model=cls.embed_model,
        )

    def test_custom_retriever(self):
        ai_filter = MetadataFilter(
            key="category", value="ai", operator=FilterOperator.EQ
        )
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=3,
            filters=MetadataFilters(filters=[ai_filter]),
        )
        nodes = retriever.retrieve("What is deep learning?")
        self.assertGreater(len(nodes), 0)

    def test_query_engine(self):
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=3)
        query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
        response = query_engine.query("Explain machine learning vs deep learning")
        self.assertTrue(len(str(response)) > 0)


# ------------------ Query & Filter Tests ------------------
class TestQueryAndFilter(VectorXTestSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.embed_model = HuggingFaceEmbedding(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        cls.vector_store = VectorXVectorStore.from_params(
            api_token=cls.vecx_api_token,
            index_name=cls.test_index_name,
            encryption_key=cls.encryption_key,
            dimension=cls.dimension,
            space_type=cls.space_type,
        )
        cls.storage_context = StorageContext.from_defaults(
            vector_store=cls.vector_store
        )
        Settings.llm = None
        cls.index = VectorStoreIndex.from_documents(
            cls.test_documents,
            storage_context=cls.storage_context,
            embed_model=cls.embed_model,
        )

    def test_basic_query(self):
        query_text = "What is Python?"
        query_embedding = self.embed_model.get_text_embedding(query_text)
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2)
        results = self.vector_store.query(query)
        self.assertGreater(len(results.nodes), 0)

    def test_filtered_query(self):
        query_text = "Explain machine learning"
        query_embedding = self.embed_model.get_text_embedding(query_text)
        ai_filter = MetadataFilter(
            key="category", value="ai", operator=FilterOperator.EQ
        )
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=2,
            filters=MetadataFilters(filters=[ai_filter]),
        )
        results = self.vector_store.query(query)
        self.assertGreater(len(results.nodes), 0)


# ------------------ Mocked VectorX Tests ------------------
class TestVectorXMock(unittest.TestCase):
    def setUp(self):
        self.mock_index = MagicMock()
        self.mock_index.dimension = 2
        self.mock_index.query.return_value = [
            {
                "id": "1",
                "similarity": 0.9,
                "meta": {"text": "mock text"},
                "vector": [0.1, 0.2],
            }
        ]
        self.store = VectorXVectorStore(
            vectorx_index=self.mock_index, dimension=2, index_name="mock_index"
        )

    def test_add_and_query_mock(self):
        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=1)
        result = self.store.query(query)
        self.assertEqual(result.similarities[0], 0.9)
        self.assertEqual(len(result.nodes), 1)

    def test_delete_non_existent_node(self):
        self.store.delete("nonexistent")

    def test_query_with_empty_filters(self):
        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2],
            similarity_top_k=1,
            filters=MetadataFilters(filters=[]),
        )
        result = self.store.query(query)
        self.assertEqual(len(result.nodes), 1)

    def test_error_on_invalid_embedding(self):
        self.store._vectorx_index = MagicMock()
        del self.store._vectorx_index.dimension
        with pytest.raises(ValueError):
            self.store.query(VectorStoreQuery(query_embedding=None, similarity_top_k=1))


# ------------------ Advanced Tests with Mocking ------------------
class TestVectorXAdvanced(unittest.TestCase):
    def setUp(self):
        self.mock_index = MagicMock()
        self.mock_index.dimension = 2
        self.mock_index.query.return_value = [
            {
                "id": "1",
                "similarity": 0.9,
                "meta": {"text": "mock text"},
                "vector": [0.1, 0.2],
            }
        ]
        self.mock_index.delete_with_filter = MagicMock()
        self.store = VectorXVectorStore(
            vectorx_index=self.mock_index, dimension=2, index_name="mock_index"
        )

    def test_initialize_vectorx_index_import_error(self):
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "vecx.vectorx":
                raise ImportError("No module named vecx.vectorx")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        with pytest.raises(ImportError):
            VectorXVectorStore._initialize_vectorx_index(
                api_token="token", encryption_key="key", index_name="idx", dimension=2
            )
        builtins.__import__ = original_import

    def test_query_hybrid_with_alpha(self):
        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2], similarity_top_k=1, mode="HYBRID", alpha=2.0
        )
        result = self.store.query(query)
        self.assertEqual(result.similarities[0], 0.9)

    def test_delete_with_error_logging(self):
        self.store._vectorx_index.delete_with_filter.side_effect = Exception(
            "Delete failed"
        )
        self.store.delete("ref_doc")

    def test_query_missing_dimension_and_no_embedding(self):
        self.store._vectorx_index = MagicMock()
        del self.store._vectorx_index.dimension
        self.store._vectorx_index.describe.side_effect = Exception("No dimension")
        with pytest.raises(ValueError):
            self.store.query(VectorStoreQuery(query_embedding=None, similarity_top_k=1))

    def test_mocked_vectorx_index_usage(self):
        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=1)
        result = self.store.query(query)
        self.assertEqual(result.similarities[0], 0.9)
        self.assertEqual(len(result.nodes), 1)


# ------------------ Run Tests ------------------
if __name__ == "__main__":
    unittest.main()
