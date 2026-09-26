def test_weaviate_vector_store_imports_with_current_weaviate_client():
    from llama_index.vector_stores.weaviate import WeaviateVectorStore

    assert WeaviateVectorStore is not None
