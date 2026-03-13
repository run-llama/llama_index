import unittest
import sys
from pathlib import Path

# Add project root to sys.path for imports

from tests.setup_class import EndeeTestSetup
from llama_index.vector_stores.endee import EndeeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    VectorStoreQuery,
)

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None

class TestEndeeQueryAndFilter(EndeeTestSetup):
    @classmethod
    def setUpClass(cls):
        """Set up test data and create index once for all tests"""
        super().setUpClass()
        if HuggingFaceEmbedding is None:
            raise unittest.SkipTest("HuggingFaceEmbedding not available; install llama-index-embeddings-huggingface")
        cls.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        
        # Create vector store (default precision int16 since endee v0.1.17)
        cls.vector_store = EndeeVectorStore.from_params(
            api_token=cls.endee_api_token,
            index_name=cls.test_index_name,
            dimension=cls.dimension,
            space_type=cls.space_type
            # precision defaults to "int16" (changed from "float16" in endee v0.1.17)
        )
        
        # Create storage context
        cls.storage_context = StorageContext.from_defaults(vector_store=cls.vector_store)
        
        # Disable LLM to focus on vector store testing
        Settings.llm = None
        
        # Create index with documents
        cls.index = VectorStoreIndex.from_documents(
            cls.test_documents,
            storage_context=cls.storage_context,
            embed_model=cls.embed_model
        )

    def test_basic_query(self):
        """Test basic query functionality without filters"""
        try:
            # Get query embedding
            query_text = "What is Python?"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            # Create VectorStoreQuery
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=2
            )
            
            # Query directly through vector store
            results = self.vector_store.query(query)
            
            # Verify results
            self.assertIsNotNone(results)
            self.assertTrue(len(results.nodes) > 0)
            # Check if any result contains Python-related content
            python_found = any("python" in node.text.lower() for node in results.nodes)
            self.assertTrue(python_found, "No Python-related content found in results")
            
            # Test another query
            query_text = "Tell me about JavaScript"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=2
            )
            
            results = self.vector_store.query(query)
            
            self.assertIsNotNone(results)
            self.assertTrue(len(results.nodes) > 0)
            # Check if any result contains JavaScript-related content
            js_found = any("javascript" in node.text.lower() for node in results.nodes)
            self.assertTrue(js_found, "No JavaScript-related content found in results")
            
        except Exception as e:
            self.fail(f"Basic query test failed: {str(e)}")

    def test_single_filter_query(self):
        """Test querying with single metadata filter"""
        try:
            # Test AI category filter
            query_text = "What is machine learning?"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            ai_filter = MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ)
            ai_filters = MetadataFilters(filters=[ai_filter])
            
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=2,
                filters=ai_filters
            )
            
            results = self.vector_store.query(query)
            
            self.assertIsNotNone(results)
            self.assertTrue(len(results.nodes) > 0)
            # Verify all results are from AI category
            all_ai = all("ai" in str(node.metadata.get("category", "")).lower() for node in results.nodes)
            self.assertTrue(all_ai, "Found results not from AI category")
            
            # Test programming category filter
            query_text = "What programming languages are mentioned?"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            prog_filter = MetadataFilter(key="category", value="programming", operator=FilterOperator.EQ)
            prog_filters = MetadataFilters(filters=[prog_filter])
            
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=3,
                filters=prog_filters
            )
            
            results = self.vector_store.query(query)
            
            self.assertIsNotNone(results)
            self.assertTrue(len(results.nodes) > 0)
            # Verify all results are from programming category
            all_prog = all("programming" in str(node.metadata.get("category", "")).lower() for node in results.nodes)
            self.assertTrue(all_prog, "Found results not from programming category")
            
        except Exception as e:
            self.fail(f"Single filter query test failed: {str(e)}")

    def test_multiple_filters_query(self):
        """Test querying with multiple metadata filters"""
        try:
            # Test category AND difficulty filter
            query_text = "What programming language is good for beginners?"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            # Python is the only programming language marked as beginner
            category_filter = MetadataFilter(key="category", value="programming", operator=FilterOperator.EQ)
            difficulty_filter = MetadataFilter(key="difficulty", value="beginner", operator=FilterOperator.EQ)
            
            complex_filters = MetadataFilters(filters=[category_filter, difficulty_filter])
            
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=2,
                filters=complex_filters
            )
            
            results = self.vector_store.query(query)
            self.assertIsNotNone(results)
            self.assertTrue(len(results.nodes) > 0)
            # Verify results match both filters
            for node in results.nodes:
                self.assertIn("programming", str(node.metadata.get("category", "")).lower())
                self.assertIn("beginner", str(node.metadata.get("difficulty", "")).lower())
            
            # Test database category with vector type
            query_text = "What database is good for similarity search?"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            db_filter = MetadataFilter(key="category", value="database", operator=FilterOperator.EQ)
            type_filter = MetadataFilter(key="type", value="vector", operator=FilterOperator.EQ)
            
            db_filters = MetadataFilters(filters=[db_filter, type_filter])
            
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=2,
                filters=db_filters
            )
            
            results = self.vector_store.query(query)
            
            self.assertIsNotNone(results)
            self.assertTrue(len(results.nodes) > 0)
            # Verify results match both filters
            for node in results.nodes:
                self.assertIn("database", str(node.metadata.get("category", "")).lower())
                self.assertIn("vector", str(node.metadata.get("type", "")).lower())
            
        except Exception as e:
            self.fail(f"Multiple filters query test failed: {str(e)}")

    def test_filter_operators(self):
        """Test filter operators supported by base.py: EQ ($eq), IN ($in), LT/LTE ($range). Not supported: NE, GT, GTE."""
        # Test unsupported NE operator - should raise ValueError
        query_text = "What advanced topics are covered?"
        query_embedding = self.embed_model.get_text_embedding(query_text)
        ne_filter = MetadataFilter(key="difficulty", value="beginner", operator=FilterOperator.NE)
        ne_filters = MetadataFilters(filters=[ne_filter])
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=3,
            filters=ne_filters,
        )
        with self.assertRaises(ValueError) as ctx:
            self.vector_store.query(query)
        self.assertIn("Unsupported filter operator", str(ctx.exception))

        # Test IN operator (supported by base.py)
        try:
            query_text = "What web technologies are discussed?"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            in_filter = MetadataFilter(
                key="language",
                value=["python", "javascript"],
                operator=FilterOperator.IN,
            )
            in_filters = MetadataFilters(filters=[in_filter])
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=3,
                filters=in_filters,
            )
            results = self.vector_store.query(query)
            self.assertIsNotNone(results)
            self.assertTrue(len(results.nodes) > 0)
            valid_langs = all(
                any(lang in str(node.metadata.get("language", "")).lower() for lang in ["python", "javascript"])
                for node in results.nodes
            )
            self.assertTrue(valid_langs, "Found results with languages other than Python or JavaScript")
        except Exception as e:
            self.fail(f"Filter operators IN test failed: {str(e)}")

    def test_invalid_filters(self):
        """Test error handling for invalid filters"""
        try:
            # Test with non-existent metadata key
            query_text = "What will I get?"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            invalid_filter = MetadataFilter(key="non_existent", value="something", operator=FilterOperator.EQ)
            invalid_filters = MetadataFilters(filters=[invalid_filter])
            
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=2,
                filters=invalid_filters
            )
            
            results = self.vector_store.query(query)
            
            # Should return empty or no results, but shouldn't crash
            self.assertIsNotNone(results)
            # Optionally verify it returns empty results
            self.assertEqual(len(results.nodes), 0, "Expected no results for invalid filter")
            
        except Exception as e:
            self.fail(f"Invalid filters test failed: {str(e)}")

    def test_hybrid_query_with_filter(self):
        """Test hybrid store query with metadata filter (hybrid=True, sparse + dense)."""
        try:
            import time
            unique_index_name = f"{self.test_index_name}_hybrid_filter_{int(time.time())}"
            self.test_indexes.add(unique_index_name)
            vector_store = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=unique_index_name,
                dimension=self.dimension,
                space_type=self.space_type,
                hybrid=True,
                sparse_dim=30522,
                model_name="splade_pp",
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            VectorStoreIndex.from_documents(
                self.test_documents[:3],
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
            query_text = "Python programming"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            category_filter = MetadataFilter(key="category", value="programming", operator=FilterOperator.EQ)
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=2,
                filters=MetadataFilters(filters=[category_filter]),
            )
            results = vector_store.query(query, hybrid=True)
            self.assertIsNotNone(results)
            self.assertGreaterEqual(len(results.nodes), 0)
        except ImportError as e:
            self.skipTest(f"Hybrid dependencies not available: {e}")
        except Exception as e:
            self.fail(f"Hybrid query with filter failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 