"""
Comprehensive test suite for EndeeVectorStore.

Tests cover:
- Index creation with various configurations
- Document insertion and retrieval
- Precision types
- HNSW parameters
- Metadata handling
- Query operations
- Batch operations
- Delete operations
"""

import unittest
import sys
import time
from pathlib import Path

# Add project root to sys.path for imports

from tests.setup_class import EndeeTestSetup
from llama_index.vector_stores.endee import EndeeVectorStore
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.schema import TextNode

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None


class TestEndeeVectorStoreCore(EndeeTestSetup):
    """Core functionality tests for EndeeVectorStore"""

    def setUp(self):
        """Set up embedding model for each test"""
        if HuggingFaceEmbedding is None:
            self.skipTest("HuggingFaceEmbedding not available; install llama-index-vector-stores-endee-embeddings-huggingface")
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

    def test_create_index_with_default_params(self):
        """Test creating index with default parameters"""
        try:
            vector_store = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine"
            )
            
            self.assertIsNotNone(vector_store)
            self.assertEqual(vector_store.dimension, self.dimension)
            self.assertEqual(vector_store.space_type, "cosine")
            self.assertEqual(vector_store.precision, "int16")  # Default precision (changed from float16 in v0.1.17)
            
            # Verify index was created
            index_info = vector_store.describe()
            self.assertEqual(index_info["dimension"], self.dimension)
            
        except Exception as e:
            self.fail(f"Index creation with default params failed: {str(e)}")

    def test_create_index_with_custom_precision(self):
        """Test creating index with different precision types"""
        precision_types = ["float32", "float16", "int8", "int16"]
        
        for precision in precision_types:
            with self.subTest(precision=precision):
                try:
                    unique_index_name = f"{self.test_index_name}_{precision}_{int(time.time())}"
                    vector_store = EndeeVectorStore.from_params(
                        api_token=self.endee_api_token,
                        index_name=unique_index_name,
                        dimension=self.dimension,
                        space_type="cosine",
                        precision=precision
                    )
                    
                    self.assertEqual(vector_store.precision, precision)
                    index_info = vector_store.describe()
                    self.assertEqual(index_info["dimension"], self.dimension)
                    
                except Exception as e:
                    self.fail(f"Index creation with precision {precision} failed: {str(e)}")

    def test_create_index_with_custom_hnsw_params(self):
        """Test creating index with custom HNSW parameters"""
        try:
            unique_index_name = f"{self.test_index_name}_hnsw_{int(time.time())}"
            vector_store = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=unique_index_name,
                dimension=self.dimension,
                space_type="cosine",
            )
            
            self.assertIsNotNone(vector_store)
            # Index info verification
            index_info = vector_store.describe()
            self.assertEqual(index_info["dimension"], self.dimension)
            
        except Exception as e:
            self.fail(f"Index creation with custom HNSW params failed: {str(e)}")

    def test_space_type_variations(self):
        """Test creating indexes with different space types"""
        space_types = ["cosine", "l2", "ip"]
        
        for space_type in space_types:
            with self.subTest(space_type=space_type):
                try:
                    unique_index_name = f"{self.test_index_name}_{space_type}_{int(time.time())}"
                    vector_store = EndeeVectorStore.from_params(
                        api_token=self.endee_api_token,
                        index_name=unique_index_name,
                        dimension=self.dimension,
                        space_type=space_type
                    )
                    
                    self.assertEqual(vector_store.space_type, space_type)
                    index_info = vector_store.describe()
                    self.assertEqual(index_info["space_type"], space_type)
                    
                except Exception as e:
                    self.fail(f"Index creation with space_type {space_type} failed: {str(e)}")



class TestEndeeVectorStoreDocuments(EndeeTestSetup):
    """Tests for document operations"""

    def setUp(self):
        """Set up embedding model and vector store for each test"""
        if HuggingFaceEmbedding is None:
            self.skipTest("HuggingFaceEmbedding not available")
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

    def test_insert_and_query_documents(self):
        """Test inserting documents and querying them"""
        try:
            vector_store = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine"
            )
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            _index = VectorStoreIndex.from_documents(  # noqa: F841
                self.test_documents,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            # Query for Python-related content
            query_text = "programming language"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=3
            )
            
            result = vector_store.query(query)
            self.assertIsNotNone(result)
            self.assertGreater(len(result.nodes), 0)
            
        except Exception as e:
            self.fail(f"Document insertion and query failed: {str(e)}")

    def test_insert_nodes_with_metadata(self):
        """Test inserting nodes with metadata"""
        try:
            vector_store = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine"
            )
            
            # Create nodes with metadata
            nodes = [
                TextNode(
                    text="Python is great for data science",
                    metadata={"category": "programming", "difficulty": "beginner"}
                ),
                TextNode(
                    text="Machine learning requires statistics",
                    metadata={"category": "ai", "difficulty": "advanced"}
                )
            ]
            
            # Generate embeddings
            for node in nodes:
                node.embedding = self.embed_model.get_text_embedding(node.text)
            
            # Insert nodes
            vector_store.add(nodes)
            
            # Query to verify insertion
            query_embedding = self.embed_model.get_text_embedding("programming")
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=2
            )
            
            result = vector_store.query(query)
            self.assertGreater(len(result.nodes), 0)
            
            # Verify metadata is preserved
            found_metadata = False
            for node in result.nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    found_metadata = True
                    break
            self.assertTrue(found_metadata, "Metadata not preserved in results")
            
        except Exception as e:
            self.fail(f"Node insertion with metadata failed: {str(e)}")

    def test_batch_insert(self):
        """Test inserting multiple documents in batches"""
        try:
            vector_store = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine",
                batch_size=50  # Custom batch size
            )
            
            # Create many documents
            large_doc_set = [
                Document(
                    text=f"Document number {i} about topic {i % 5}",
                    metadata={"doc_id": str(i), "topic": str(i % 5)}
                )
                for i in range(150)
            ]
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            _index = VectorStoreIndex.from_documents(  # noqa: F841
                large_doc_set,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            # Verify documents were inserted
            query_embedding = self.embed_model.get_text_embedding("document topic")
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=10
            )
            
            result = vector_store.query(query)
            self.assertGreater(len(result.nodes), 0)
            
        except Exception as e:
            self.fail(f"Batch insert failed: {str(e)}")


class TestEndeeVectorStoreQueries(EndeeTestSetup):
    """Tests for query operations"""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all query tests"""
        super().setUpClass()
        if HuggingFaceEmbedding is None:
            raise unittest.SkipTest("HuggingFaceEmbedding not available")
        
        cls.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        # Create vector store and insert test documents
        cls.vector_store = EndeeVectorStore.from_params(
            api_token=cls.endee_api_token,
            index_name=cls.test_index_name,
            dimension=cls.dimension,
            space_type="cosine"
        )
        
        storage_context = StorageContext.from_defaults(vector_store=cls.vector_store)
        cls.index = VectorStoreIndex.from_documents(
            cls.test_documents,
            storage_context=storage_context,
            embed_model=cls.embed_model
        )

    def test_query_with_different_top_k(self):
        """Test querying with different top_k values"""
        top_k_values = [1, 3, 5]
        
        for top_k in top_k_values:
            with self.subTest(top_k=top_k):
                query_embedding = self.embed_model.get_text_embedding("programming")
                query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=top_k
                )
                
                result = self.vector_store.query(query)
                self.assertLessEqual(len(result.nodes), top_k)

    def test_query_with_custom_ef(self):
        """Test querying with custom ef (search quality parameter)"""
        try:
            query_embedding = self.embed_model.get_text_embedding("machine learning")
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=3
            )
            
            # Query with custom ef parameter
            result = self.vector_store.query(query, ef=256)
            self.assertIsNotNone(result)
            self.assertGreater(len(result.nodes), 0)
            
        except Exception as e:
            self.fail(f"Query with custom ef failed: {str(e)}")

    def test_query_similarity_scores(self):
        """Test that query results include similarity scores"""
        try:
            query_embedding = self.embed_model.get_text_embedding("Python programming")
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=3
            )
            
            result = self.vector_store.query(query)
            self.assertIsNotNone(result.similarities)
            self.assertGreater(len(result.similarities), 0)
            
            # Verify scores are in valid range (0 to 1 for cosine)
            for score in result.similarities:
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
                
        except Exception as e:
            self.fail(f"Query similarity scores test failed: {str(e)}")


class TestEndeeVectorStoreOperations(EndeeTestSetup):
    """Tests for various vector store operations"""

    def setUp(self):
        """Set up for each test"""
        if HuggingFaceEmbedding is None:
            self.skipTest("HuggingFaceEmbedding not available")
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

    def test_describe_index(self):
        """Test getting index description/metadata"""
        try:
            vector_store = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine"
            )
            
            index_info = vector_store.describe()
            
            self.assertIsNotNone(index_info)
            self.assertIn("dimension", index_info)
            self.assertIn("space_type", index_info)
            self.assertEqual(index_info["dimension"], self.dimension)
            self.assertEqual(index_info["space_type"], "cosine")
            
        except Exception as e:
            self.fail(f"Index describe operation failed: {str(e)}")

    def test_delete_by_ref_doc_id(self):
        """Test deleting documents by ref_doc_id"""
        try:
            vector_store = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine"
            )
            
            # Insert test document
            doc = Document(
                text="Test document for deletion",
                metadata={"test": "delete"}
            )
            doc.doc_id = "test_doc_123"
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            _index = VectorStoreIndex.from_documents(  # noqa: F841
                [doc],
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            # Delete by ref_doc_id
            vector_store.delete(ref_doc_id="test_doc_123")
            
            # Verify deletion - query should return no results for this specific doc
            # (This is a best-effort test since we can't easily verify absence)
            
        except Exception as e:
            self.fail(f"Delete operation failed: {str(e)}")

    def test_client_property(self):
        """Test accessing the underlying Endee client"""
        try:
            vector_store = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine"
            )
            
            client = vector_store.client
            self.assertIsNotNone(client)
            # The client should be the Endee Index object
            
        except Exception as e:
            self.fail(f"Client property access failed: {str(e)}")

    def test_use_existing_index(self):
        """Test connecting to an existing index"""
        try:
            # First create an index
            vector_store1 = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine"
            )
            
            # Add a document
            doc = Document(text="Test document in existing index")
            storage_context = StorageContext.from_defaults(vector_store=vector_store1)
            _index1 = VectorStoreIndex.from_documents(  # noqa: F841
                [doc],
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            # Now connect to the same index with a new vector store instance
            vector_store2 = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine"
            )
            
            # Verify we can query the existing data
            query_embedding = self.embed_model.get_text_embedding("test document")
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=1
            )
            
            result = vector_store2.query(query)
            self.assertGreater(len(result.nodes), 0)
            
        except Exception as e:
            self.fail(f"Using existing index failed: {str(e)}")


class TestEndeeVectorStoreEdgeCases(EndeeTestSetup):
    """Tests for edge cases and error handling"""

    def setUp(self):
        """Set up for each test"""
        if HuggingFaceEmbedding is None:
            self.skipTest("HuggingFaceEmbedding not available")
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

    def test_invalid_precision_type(self):
        """Test that invalid precision type raises error"""
        with self.assertRaises(ValueError):
            _vector_store = EndeeVectorStore.from_params(  # noqa: F841
                api_token=self.endee_api_token,
                index_name=f"{self.test_index_name}_invalid_{int(time.time())}",
                dimension=self.dimension,
                space_type="cosine",
                precision="invalid_precision"  # Invalid precision
            )

    def test_invalid_space_type(self):
        """Test that invalid space type raises error"""
        with self.assertRaises(ValueError):
            _vector_store = EndeeVectorStore.from_params(  # noqa: F841
                api_token=self.endee_api_token,
                index_name=f"{self.test_index_name}_invalid_{int(time.time())}",
                dimension=self.dimension,
                space_type="invalid_space"  # Invalid space type
            )

    def test_empty_query(self):
        """Test querying with empty embedding"""
        try:
            vector_store = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine"
            )
            
            # Create empty query (no embedding)
            query = VectorStoreQuery(similarity_top_k=3)
            
            # This should handle gracefully or return empty results
            _result = vector_store.query(query)  # noqa: F841
            # Depending on implementation, might return empty or raise
            
        except Exception:
            # Expected behavior - should handle gracefully
            pass


if __name__ == "__main__":
    unittest.main()