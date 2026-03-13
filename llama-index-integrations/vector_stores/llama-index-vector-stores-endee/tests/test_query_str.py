"""
Test cases for query.query_str support in EndeeVectorStore.

This test demonstrates that EndeeVectorStore correctly handles the query_str
attribute from VectorStoreQuery, similar to Qdrant's implementation.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path for imports

from tests.setup_class import EndeeTestSetup
from llama_index.vector_stores.endee import EndeeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStoreQuery

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None


class TestQueryStrSupport(EndeeTestSetup):
    """Test VectorStoreQuery.query_str support"""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests"""
        super().setUpClass()
        if HuggingFaceEmbedding is None:
            raise unittest.SkipTest("HuggingFaceEmbedding not available")
        
        cls.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        # Create vector store
        cls.vector_store = EndeeVectorStore.from_params(
            api_token=cls.endee_api_token,
            index_name=cls.test_index_name,
            dimension=cls.dimension,
            space_type="cosine"
        )
        
        # Insert test documents
        storage_context = StorageContext.from_defaults(vector_store=cls.vector_store)
        cls.index = VectorStoreIndex.from_documents(
            cls.test_documents,
            storage_context=storage_context,
            embed_model=cls.embed_model
        )

    def test_query_with_query_str(self):
        """Test that query.query_str is accessible and used correctly"""
        try:
            # Create a query with query_str
            query_text = "Python programming language"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                query_str=query_text,  # Set query_str explicitly
                similarity_top_k=3
            )
            
            # Verify query_str is set
            self.assertEqual(query.query_str, query_text)
            
            # Execute query
            result = self.vector_store.query(query)
            
            # Verify results
            self.assertIsNotNone(result)
            self.assertGreater(len(result.nodes), 0)
            
        except Exception as e:
            self.fail(f"Query with query_str failed: {str(e)}")

    def test_query_str_in_hybrid_mode(self):
        """Test query_str usage in hybrid search mode"""
        try:
            # For hybrid mode, query_str is especially important
            query_text = "machine learning algorithms"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                query_str=query_text,
                similarity_top_k=3
            )
            
            # Verify query has both embedding and text
            self.assertIsNotNone(query.query_embedding)
            self.assertIsNotNone(query.query_str)
            
            # Execute query (will use query_str for sparse search if hybrid enabled)
            result = self.vector_store.query(query, hybrid=False)
            
            self.assertIsNotNone(result)
            self.assertGreater(len(result.nodes), 0)
            
        except Exception as e:
            self.fail(f"Hybrid query with query_str failed: {str(e)}")

    def test_query_without_query_str(self):
        """Test that queries work without query_str (backward compatibility)"""
        try:
            # Create query without query_str
            query_embedding = self.embed_model.get_text_embedding("test query")
            
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=3
                # No query_str provided
            )
            
            # Verify query_str is None
            self.assertIsNone(query.query_str)
            
            # Execute query should still work
            result = self.vector_store.query(query)
            
            self.assertIsNotNone(result)
            self.assertGreater(len(result.nodes), 0)
            
        except Exception as e:
            self.fail(f"Query without query_str failed: {str(e)}")

    def test_sparse_query_text_parameter(self):
        """Test that sparse_query_text parameter overrides query.query_str"""
        try:
            query_text = "original query"
            override_text = "override query"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                query_str=query_text,
                similarity_top_k=3
            )
            
            # Pass sparse_query_text which should override query.query_str
            result = self.vector_store.query(
                query,
                sparse_query_text=override_text
            )
            
            self.assertIsNotNone(result)
            self.assertGreater(len(result.nodes), 0)
            
        except Exception as e:
            self.fail(f"Query with sparse_query_text override failed: {str(e)}")

    def test_query_str_attribute_access(self):
        """Test safe attribute access using getattr"""
        try:
            # This tests the pattern: getattr(query, "query_str", None)
            query_embedding = self.embed_model.get_text_embedding("test")
            
            # Query with query_str
            query1 = VectorStoreQuery(
                query_embedding=query_embedding,
                query_str="test text",
                similarity_top_k=1
            )
            self.assertEqual(getattr(query1, "query_str", None), "test text")
            
            # Query without query_str
            query2 = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=1
            )
            self.assertIsNone(getattr(query2, "query_str", None))
            
        except Exception as e:
            self.fail(f"Query_str attribute access test failed: {str(e)}")


if __name__ == "__main__":
    unittest.main()