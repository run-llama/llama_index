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

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

class TestCustomRetrieval(EndeeTestSetup):
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

    def test_custom_retriever(self):
        """Test custom retriever functionality with filters"""
        try:
            # Create AI category filter
            ai_filter = MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ)
            ai_filters = MetadataFilters(filters=[ai_filter])
            
            # Create custom retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3,
                filters=ai_filters
            )
            
            # Test retrieval
            nodes = retriever.retrieve("What is deep learning?")
            
            # Verify results
            self.assertIsNotNone(nodes)
            self.assertTrue(len(nodes) > 0)
            self.assertLessEqual(len(nodes), 3)  # Should not exceed top_k
            
            # Verify all results are from AI category and contain scores
            for node in nodes:
                self.assertIsNotNone(node.score)
                self.assertIn("ai", str(node.node.metadata.get("category", "")).lower())
                self.assertIsNotNone(node.node.text)
            
            # Test with different query
            nodes = retriever.retrieve("Explain machine learning concepts")
            
            self.assertIsNotNone(nodes)
            self.assertTrue(len(nodes) > 0)
            self.assertLessEqual(len(nodes), 3)
            
            # Verify results contain ML-related content
            ml_found = any("machine learning" in node.node.text.lower() for node in nodes)
            self.assertTrue(ml_found, "No machine learning content found in results")
            
        except Exception as e:
            self.fail(f"Custom retriever test failed: {str(e)}")

    def test_custom_query_engine(self):
        """Test query engine with custom retriever"""
        try:
            # Create custom retriever with AI filter
            ai_filter = MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ)
            ai_filters = MetadataFilters(filters=[ai_filter])
            
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3,
                filters=ai_filters
            )
            
            # Create query engine
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                verbose=True
            )
            
            # Test querying
            response = query_engine.query("Explain the difference between machine learning and deep learning")
            
            # Verify response
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'response'))  # Should have a response attribute
            self.assertTrue(len(str(response)) > 0)  # Response should not be empty
            
            # Test source nodes
            source_nodes = response.source_nodes
            self.assertIsNotNone(source_nodes)
            self.assertTrue(len(source_nodes) > 0)
            
            # Verify source nodes are from AI category
            for node in source_nodes:
                self.assertIn("ai", str(node.metadata.get("category", "")).lower())
            
        except Exception as e:
            self.fail(f"Custom query engine test failed: {str(e)}")

    def test_direct_vectorstore_querying(self):
        """Test direct vector store querying with embeddings"""
        try:
            # Test querying for database-related content
            query_text = "What are vector databases?"
            query_embedding = self.embed_model.get_text_embedding(query_text)
            
            # Create database category filter
            db_filter = MetadataFilter(key="category", value="database", operator=FilterOperator.EQ)
            db_filters = MetadataFilters(filters=[db_filter])
            
            # Create vector store query
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=2,
                filters=db_filters
            )
            
            # Execute query
            query_result = self.vector_store.query(vector_store_query)
            
            # Verify results
            self.assertIsNotNone(query_result)
            self.assertTrue(len(query_result.nodes) > 0)
            self.assertLessEqual(len(query_result.nodes), 2)  # Should not exceed top_k
            
            # Verify similarities are present
            self.assertIsNotNone(query_result.similarities)
            self.assertEqual(len(query_result.nodes), len(query_result.similarities))
            
            # Check each result (base.py returns similarity from backend; range depends on space_type)
            for node, score in zip(query_result.nodes, query_result.similarities):
                self.assertIn("database", str(node.metadata.get("category", "")).lower())
                self.assertIsInstance(score, (int, float))
                
            # Test with vector database specific content
            vector_db_found = any("vector database" in node.text.lower() for node in query_result.nodes)
            self.assertTrue(vector_db_found, "No vector database content found in results")
            
        except Exception as e:
            self.fail(f"Direct vector store querying test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 