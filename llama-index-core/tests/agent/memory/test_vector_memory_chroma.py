"""
    Test VectorMemory with ChromaVectorStore.

Verifies that VectorMemory.get() does not raise KeyError when the Chroma
collection contains nodes without sub_dicts metadata (e.g. document chunks
from a shared/pre-existing collection). See: KeyError 'sub_dicts' bug fix.

Requires llama-index-core with the fix (node.metadata.get('sub_dicts', [])).
"""

import chromadb
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.memory import VectorMemory
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore


def test_vector_memory_with_chroma_handles_document_nodes() -> None:
    """
    VectorMemory.get() should not raise KeyError when Chroma returns document nodes.

    When using a Chroma collection that was populated with documents (e.g. via
    VectorStoreIndex.from_documents), similarity search returns nodes without
    sub_dicts. VectorMemory must gracefully skip these instead of raising KeyError.
    """
    # Use EphemeralClient for isolated in-memory test (no disk, no cleanup needed)
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        "vector_memory_test", metadata={"hnsw:space": "cosine"}
    )

    # ChromaVectorStore with flat_metadata=False to allow document-style metadata
    # (sub_dicts would still fail at add time for Chroma, but document nodes work)
    vector_store = ChromaVectorStore(chroma_collection=collection, flat_metadata=False)

    # Add document-style nodes (no sub_dicts) - simulates shared collection
    # from VectorStoreIndex.from_documents or similar
    doc_nodes = [
        TextNode(
            text="Document content about apples",
            metadata={"file_name": "doc1.pdf", "page_label": "1"},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        ),
        TextNode(
            text="Document content about oranges",
            metadata={"file_name": "doc2.pdf", "page_label": "2"},
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
        ),
    ]
    vector_store.add(doc_nodes)

    # Create VectorMemory with this Chroma-backed store
    embed_model = MockEmbedding(embed_dim=5)
    vector_memory = VectorMemory.from_defaults(
        vector_store=vector_store,
        embed_model=embed_model,
        retriever_kwargs={"similarity_top_k": 3},
    )

    # Act - should NOT raise KeyError: 'sub_dicts'
    # Retrieved nodes are document chunks without sub_dicts; they get skipped
    msgs = vector_memory.get("apples and oranges")

    # Assert - returns empty list (no VectorMemory-formatted nodes in collection)
    assert isinstance(msgs, list)
    assert len(msgs) == 0
