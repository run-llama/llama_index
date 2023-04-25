from typing import Any, Dict, List, cast
from unittest.mock import patch
from gpt_index.data_structs.node_v2 import DocumentRelationship, Node
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.vector_store.base import GPTVectorStoreIndex
from gpt_index.readers.schema.base import Document
from gpt_index.storage.storage_context import StorageContext
from gpt_index.vector_stores.simple import SimpleVectorStore
from tests.mock_utils.mock_decorator import patch_common
from tests.indices.vector_store.test_simple import (
    mock_get_query_embedding,
    mock_get_text_embedding,
    mock_get_text_embeddings,
)


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_faiss_query(
    _mock_query_embed: Any,
    _mock_texts_embed: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    faiss_storage_context: StorageContext,
) -> None:
    """Test embedding query."""
    index = GPTVectorStoreIndex.from_documents(
        documents=documents, storage_context=faiss_storage_context
    )

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 1
    assert nodes[0].node.text == "This is another test."


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_simple_query(
    _mock_query_embed: Any,
    _mock_text_embeds: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test embedding query."""
    index = GPTVectorStoreIndex.from_documents(documents)

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 1
    assert nodes[0].node.text == "This is another test."


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "_get_query_embedding", side_effect=mock_get_query_embedding
)
def test_query_and_similarity_scores(
    _mock_query_embed: Any,
    _mock_text_embeds: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
) -> None:
    """Test that sources nodes have similarity scores."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    index = GPTVectorStoreIndex.from_documents([document])

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) > 0
    assert nodes[0].score is not None


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_simple_check_ids(
    _mock_query_embed: Any,
    _mock_text_embeds: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test build GPTSimpleVectorIndex."""
    ref_doc_id = "ref_doc_id_test"
    source_rel = {DocumentRelationship.SOURCE: ref_doc_id}
    all_nodes = [
        Node("Hello world.", doc_id="node1", relationships=source_rel),
        Node("This is a test.", doc_id="node2", relationships=source_rel),
        Node("This is another test.", doc_id="node3", relationships=source_rel),
        Node("This is a test v2.", doc_id="node4", relationships=source_rel),
    ]
    index = GPTVectorStoreIndex(all_nodes)

    # test query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert nodes[0].node.text == "This is another test."
    assert nodes[0].node.ref_doc_id == "ref_doc_id_test"
    assert nodes[0].node.doc_id == "node3"
    vector_store = cast(SimpleVectorStore, index._vector_store)
    assert "node3" in vector_store._data.embedding_dict
    assert "node3" in vector_store._data.text_id_to_doc_id


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_faiss_check_ids(
    _mock_query_embed: Any,
    _mock_texts_embed: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    faiss_storage_context: StorageContext,
) -> None:
    """Test embedding query."""

    ref_doc_id = "ref_doc_id_test"
    source_rel = {DocumentRelationship.SOURCE: ref_doc_id}
    all_nodes = [
        Node("Hello world.", doc_id="node1", relationships=source_rel),
        Node("This is a test.", doc_id="node2", relationships=source_rel),
        Node("This is another test.", doc_id="node3", relationships=source_rel),
        Node("This is a test v2.", doc_id="node4", relationships=source_rel),
    ]

    index = GPTVectorStoreIndex(all_nodes, storage_context=faiss_storage_context)

    # test query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert nodes[0].node.text == "This is another test."
    assert nodes[0].node.ref_doc_id == "ref_doc_id_test"
    assert nodes[0].node.doc_id == "node3"


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "_get_query_embedding", side_effect=mock_get_query_embedding
)
def test_query_and_count_tokens(
    _mock_query_embed: Any,
    _mock_text_embeds: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
) -> None:
    """Test embedding query."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    index = GPTVectorStoreIndex.from_documents([document])
    assert index.service_context.embed_model.total_tokens_used == 20

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    _ = retriever.retrieve(QueryBundle(query_str))
    assert index.service_context.embed_model.last_token_usage == 3
