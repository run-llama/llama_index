import sys
from typing import Any, Dict, List, cast
from unittest.mock import MagicMock, patch
from gpt_index.data_structs.node_v2 import DocumentRelationship, Node
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.vector_store.vector_indices import (
    GPTFaissIndex,
    GPTSimpleVectorIndex,
)
from gpt_index.readers.schema.base import Document
from gpt_index.vector_stores.simple import SimpleVectorStore
from tests.indices.vector_store.test_base import MockFaissIndex
from tests.mock_utils.mock_decorator import patch_common
from tests.indices.vector_store.test_base import (
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
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    # NOTE: mock faiss import
    sys.modules["faiss"] = MagicMock()
    # NOTE: mock faiss index
    faiss_index = MockFaissIndex()

    index_kwargs, retrieval_kwargs = struct_kwargs
    index = GPTFaissIndex.from_documents(
        documents=documents, faiss_index=faiss_index, **index_kwargs
    )

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(**retrieval_kwargs)
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
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    index_kwargs, retrieval_kwargs = struct_kwargs
    index = GPTSimpleVectorIndex.from_documents(documents, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(**retrieval_kwargs)
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
    struct_kwargs: Dict,
) -> None:
    """Test that sources nodes have similarity scores."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    index_kwargs, retrieval_kwargs = struct_kwargs
    index = GPTSimpleVectorIndex.from_documents([document], **index_kwargs)

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(**retrieval_kwargs)
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
    struct_kwargs: Dict,
) -> None:
    """Test build GPTSimpleVectorIndex."""
    index_kwargs, retrieval_kwargs = struct_kwargs

    ref_doc_id = "ref_doc_id_test"
    source_rel = {DocumentRelationship.SOURCE: ref_doc_id}
    all_nodes = [
        Node("Hello world.", doc_id="node1", relationships=source_rel),
        Node("This is a test.", doc_id="node2", relationships=source_rel),
        Node("This is another test.", doc_id="node3", relationships=source_rel),
        Node("This is a test v2.", doc_id="node4", relationships=source_rel),
    ]
    index = GPTSimpleVectorIndex(all_nodes, **index_kwargs)

    # test query
    query_str = "What is?"
    retriever = index.as_retriever(**retrieval_kwargs)
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
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    # NOTE: mock faiss import
    sys.modules["faiss"] = MagicMock()
    # NOTE: mock faiss index
    faiss_index = MockFaissIndex()

    index_kwargs, retrieval_kwargs = struct_kwargs

    ref_doc_id = "ref_doc_id_test"
    source_rel = {DocumentRelationship.SOURCE: ref_doc_id}
    all_nodes = [
        Node("Hello world.", doc_id="node1", relationships=source_rel),
        Node("This is a test.", doc_id="node2", relationships=source_rel),
        Node("This is another test.", doc_id="node3", relationships=source_rel),
        Node("This is a test v2.", doc_id="node4", relationships=source_rel),
    ]

    index = GPTFaissIndex(all_nodes, faiss_index=faiss_index, **index_kwargs)

    # test query
    query_str = "What is?"
    retriever = index.as_retriever(**retrieval_kwargs)
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
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    index_kwargs, retrieval_kwargs = struct_kwargs
    index = GPTSimpleVectorIndex.from_documents([document], **index_kwargs)
    assert index.service_context.embed_model.total_tokens_used == 20

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(**retrieval_kwargs)
    _ = retriever.retrieve(QueryBundle(query_str))
    assert index.service_context.embed_model.last_token_usage == 3
