from typing import List, cast
import pytest
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.vector_stores.types import (
    VectorStoreQueryResult,
    VectorStoreQueryMode,
)
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.schema import (
    Document,
    NodeRelationship,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
    ImageNode,
)
from llama_index.core.storage.storage_context import StorageContext
from tests.indices.vector_store.conftest import MockVectorStoreGeneratesEmbeddings


def test_simple_query(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    mock_embed_model,
) -> None:
    """Test embedding query."""
    index = VectorStoreIndex.from_documents(documents, embed_model=mock_embed_model)

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "This is another test."


def test_query_and_similarity_scores(
    patch_llm_predictor,
    patch_token_text_splitter,
) -> None:
    """Test that sources nodes have similarity scores."""
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    document = Document(text=doc_text)
    index = VectorStoreIndex.from_documents([document])

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) > 0
    assert nodes[0].score is not None


def test_simple_check_ids(
    patch_llm_predictor,
    patch_token_text_splitter,
) -> None:
    """Test build VectorStoreIndex."""
    ref_doc_id = "ref_doc_id_test"
    source_rel = {NodeRelationship.SOURCE: RelatedNodeInfo(node_id=ref_doc_id)}
    all_nodes = [
        TextNode(text="Hello world.", id_="node1", relationships=source_rel),
        TextNode(text="This is a test.", id_="node2", relationships=source_rel),
        TextNode(text="This is another test.", id_="node3", relationships=source_rel),
        TextNode(text="This is a test v2.", id_="node4", relationships=source_rel),
    ]
    index = VectorStoreIndex(all_nodes)

    # test query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert nodes[0].node.get_content() == "This is another test."
    assert nodes[0].node.ref_doc_id == "ref_doc_id_test"
    assert nodes[0].node.node_id == "node3"
    vector_store = cast(SimpleVectorStore, index._vector_store)
    assert "node3" in vector_store._data.embedding_dict
    assert "node3" in vector_store._data.text_id_to_ref_doc_id


def test_query(
    patch_llm_predictor,
    patch_token_text_splitter,
) -> None:
    """Test embedding query."""
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    document = Document(text=doc_text)
    index = VectorStoreIndex.from_documents([document])

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    _ = retriever.retrieve(QueryBundle(query_str))


def test_query_image_node() -> None:
    """Test embedding query."""
    image_node = ImageNode(
        image="potato", embeddings=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    text_node = TextNode(
        text="potato", embeddings=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    index = VectorStoreIndex.from_documents([])
    index.insert_nodes([image_node, text_node])

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    results = retriever.retrieve(QueryBundle(query_str))

    assert len(results) == 2

    text_node = next(
        node
        for node in results
        if isinstance(node.node, TextNode) and not isinstance(node.node, ImageNode)
    )
    image_node = next(node for node in results if isinstance(node.node, ImageNode))

    assert image_node.node.node_id == image_node.node_id
    assert isinstance(image_node.node, ImageNode)
    assert image_node.node.image == "potato"
    assert text_node.node.node_id == text_node.node_id
    assert isinstance(text_node.node, TextNode)
    assert text_node.node.text == "potato"


def test_insert_fetched_nodes_handles_all_branches():
    """Test _insert_fetched_nodes_into_query_result for full branch coverage."""
    fetched_nodes = [
        TextNode(id_="0", text="doc 0"),
        TextNode(id_="1", text="doc 1"),
        TextNode(id_="two", text="doc two"),
    ]

    query_result = VectorStoreQueryResult(
        ids=[0, "1", "unknown"], similarities=[0.9, 0.8, 0.7], nodes=None
    )

    dummy_index = VectorStoreIndex([])

    retriever = VectorIndexRetriever(
        index=dummy_index, vector_store=None, docstore=None, embed_model=None
    )

    with pytest.raises(KeyError) as exc_info:
        retriever._insert_fetched_nodes_into_query_result(query_result, fetched_nodes)

    assert "Node ID 0 not found in index." in str(exc_info.value)


def test_insert_fetched_nodes_with_nodes_present():
    """Test _insert_fetched_nodes_into_query_result with `nodes` present instead of `ids`."""
    fetched_nodes = [TextNode(id_="abc", text="Updated text")]

    # This simulates query_result.nodes populated with old version of the same node
    old_node = TextNode(id_="abc", text="Old text")

    query_result = VectorStoreQueryResult(nodes=[old_node], similarities=[0.9])

    dummy_index = VectorStoreIndex([])

    retriever = VectorIndexRetriever(
        index=dummy_index, vector_store=None, docstore=None, embed_model=None
    )

    new_nodes = retriever._insert_fetched_nodes_into_query_result(
        query_result, fetched_nodes
    )

    # Should have replaced old node with the new one
    assert len(new_nodes) == 1
    assert new_nodes[0].text == "Updated text"


def test_generates_embeddings_false_default(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that generates_embeddings defaults to False."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )

    # Default vector store should not generate embeddings
    assert index._vector_store.generates_embeddings is False
    # Embed model should be initialized
    assert index._embed_model is not None


def test_generates_embeddings_true_skips_client_embedding(
    documents: List[Document],
) -> None:
    """Test that generates_embeddings=True prevents client-side embedding."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    # When generates_embeddings=True, we don't need to provide embed_model
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    # Vector store claims to generate embeddings
    assert index._vector_store.generates_embeddings is True
    # No embed model should be created when vector store generates embeddings
    assert index._embed_model is None

    # Verify nodes are added without embeddings (vector store will handle it)
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_retriever_needs_embedding_false_when_generates_embeddings(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that _needs_embedding returns False when generates_embeddings=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
    )

    retriever = index.as_retriever()
    # When vector store generates embeddings, retriever should not need to embed
    assert retriever._needs_embedding() is False


def test_retriever_needs_embedding_true_when_no_generates_embeddings(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that _needs_embedding returns True for normal vector store."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )

    retriever = index.as_retriever()
    # Normal vector store requires client-side embedding
    assert retriever._needs_embedding() is True


def test_retriever_needs_embedding_false_with_text_search_mode(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that TEXT_SEARCH mode doesn't need embedding regardless."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )

    retriever = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.TEXT_SEARCH
    )
    # TEXT_SEARCH doesn't need embeddings even with normal vector store
    assert retriever._needs_embedding() is False


def test_retriever_needs_embedding_false_with_sparse_mode(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that SPARSE mode doesn't need embedding regardless."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )

    retriever = index.as_retriever(vector_store_query_mode=VectorStoreQueryMode.SPARSE)
    # SPARSE doesn't need embeddings even with normal vector store
    assert retriever._needs_embedding() is False


def test_retriever_needs_embedding_true_with_hybrid_mode(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that HYBRID mode needs embedding for normal vector store."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )

    retriever = index.as_retriever(vector_store_query_mode=VectorStoreQueryMode.HYBRID)
    # HYBRID mode needs embeddings with normal vector store
    assert retriever._needs_embedding() is True


def test_embed_model_none_when_generates_embeddings(
    documents: List[Document],
) -> None:
    """Test that embed_model is None when vector store generates embeddings."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    # When vector store generates embeddings, no embed model should be created
    assert index._embed_model is None


def test_embed_model_initialized_without_generates_embeddings(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test that embed_model is initialized for normal vector stores."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )

    # Normal vector store requires embed model
    assert index._embed_model is not None
    assert index._embed_model is mock_embed_model


def test_generates_embeddings_with_no_embed_model_provided(
    documents: List[Document],
) -> None:
    """Test that generates_embeddings works without providing embed_model."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    # Should not raise error even without embed_model
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    assert index._embed_model is None
    assert index._vector_store.generates_embeddings is True


def test_skip_embedding_with_generates_embeddings_both_true(
    documents: List[Document],
) -> None:
    """Test that both skip_embedding=True and generates_embeddings=True work together."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        skip_embedding=True,
    )

    # Both flags set to True
    assert index._skip_embedding is True
    assert index._vector_store.generates_embeddings is True
    # No embed model should be created
    assert index._embed_model is None

    # Verify embeddings are empty
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_skip_embedding_true_with_normal_vector_store(
    documents: List[Document],
    mock_embed_model,
) -> None:
    """Test skip_embedding=True with a vector store that generates embeddings disabled."""
    from tests.indices.multi_modal.conftest import MockVectorStoreWithSkipEmbedding

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreWithSkipEmbedding()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
        skip_embedding=True,
    )

    # skip_embedding=True, generates_embeddings=False (default)
    assert index._skip_embedding is True
    assert index._vector_store.generates_embeddings is False
    # Embed model is still initialized but not used for embedding
    assert index._embed_model is not None

    # Verify embeddings are cleared by skip_embedding
    for embedding in index._vector_store.data.embedding_dict.values():
        assert embedding == []


def test_node_embeddings_cleared_with_generates_embeddings() -> None:
    """Test that nodes without embeddings work with generates_embeddings=True."""
    # Create nodes without embeddings (as they would come from text documents)
    nodes = [
        TextNode(text="Hello world."),
        TextNode(text="This is a test."),
    ]

    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    # Add nodes to a vector store with generates_embeddings=True
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )

    # Vector store should receive nodes, embeddings handling is vector store's job
    assert len(index._vector_store.data.embedding_dict) == 2


def test_text_nodes_with_generates_embeddings(
    documents: List[Document],
) -> None:
    """Test TextNode handling with generates_embeddings=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    assert len(index.index_struct.nodes_dict) == len(documents)
    # All nodes should be in vector store
    assert len(index._vector_store.data.embedding_dict) == len(documents)


def test_default_query_mode_with_generates_embeddings(
    documents: List[Document],
) -> None:
    """Test DEFAULT query mode with generates_embeddings=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    retriever = index.as_retriever(vector_store_query_mode=VectorStoreQueryMode.DEFAULT)
    # DEFAULT mode with generates_embeddings doesn't need embedding
    assert retriever._needs_embedding() is False


def test_text_search_mode_with_generates_embeddings(
    documents: List[Document],
) -> None:
    """Test TEXT_SEARCH mode with generates_embeddings=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    retriever = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.TEXT_SEARCH
    )
    # TEXT_SEARCH never needs embedding regardless
    assert retriever._needs_embedding() is False


def test_sparse_mode_with_generates_embeddings(
    documents: List[Document],
) -> None:
    """Test SPARSE mode with generates_embeddings=True."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    retriever = index.as_retriever(vector_store_query_mode=VectorStoreQueryMode.SPARSE)
    # SPARSE never needs embedding regardless
    assert retriever._needs_embedding() is False


def test_query_embed_model_not_needed_with_generates_embeddings(
    documents: List[Document],
) -> None:
    """Test that embed_model is not needed when vector store generates embeddings."""
    storage_context = StorageContext.from_defaults(
        vector_store=MockVectorStoreGeneratesEmbeddings()
    )

    # Should create index without error even though embed_model is None
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    # Verify embed_model is None
    assert index._embed_model is None
    # Verify retriever doesn't try to embed when generates_embeddings=True
    retriever = index.as_retriever()
    assert retriever._needs_embedding() is False
