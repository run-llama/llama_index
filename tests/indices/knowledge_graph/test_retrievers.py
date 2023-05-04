from typing import Any, List
from unittest.mock import patch
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.indices.knowledge_graph.retrievers import KGTableRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.readers.schema.base import Document
from tests.indices.knowledge_graph.test_base import (
    MockEmbedding,
    mock_extract_triplets,
)
from tests.mock_utils.mock_prompts import MOCK_QUERY_KEYWORD_EXTRACT_PROMPT


@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_as_retriever(
    _patch_extract_triplets: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test query."""
    index = GPTKnowledgeGraphIndex.from_documents(
        documents, service_context=mock_service_context
    )
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle("foo"))
    # when include_text is True, the first node is the raw text
    # the second node is the query
    rel_initial_text = (
        "The following are knowledge triplets "
        "in the form of (subset, predicate, object):\n"
    )
    raw_text = "(foo, is, bar)"
    query = rel_initial_text + "('foo', 'is', 'bar')"
    assert len(nodes) == 2
    assert nodes[0].node.get_text() == raw_text
    assert nodes[1].node.get_text() == query


@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retrievers(
    _patch_extract_triplets: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    # test specific retriever class
    index = GPTKnowledgeGraphIndex.from_documents(
        documents, service_context=mock_service_context
    )
    retriever = KGTableRetriever(
        index,
        query_keyword_extract_template=MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
    )
    query_bundle = QueryBundle(query_str="foo", custom_embedding_strs=["foo"])
    nodes = retriever.retrieve(query_bundle)
    assert nodes[0].node.get_text() == "(foo, is, bar)"
    assert (
        nodes[1].node.get_text() == "The following are knowledge triplets in the "
        "form of (subset, predicate, object):\n('foo', 'is', 'bar')"
    )


@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retriever_no_text(
    _patch_extract_triplets: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    # test specific retriever class
    index = GPTKnowledgeGraphIndex.from_documents(
        documents, service_context=mock_service_context
    )
    retriever = KGTableRetriever(
        index,
        query_keyword_extract_template=MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
        include_text=False,
    )
    query_bundle = QueryBundle(query_str="foo", custom_embedding_strs=["foo"])
    nodes = retriever.retrieve(query_bundle)
    assert (
        nodes[0].node.get_text()
        == "The following are knowledge triplets in the form of "
        "(subset, predicate, object):\n('foo', 'is', 'bar')"
    )


@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retrieve_similarity(
    _patch_extract_triplets: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test query."""
    mock_service_context.embed_model = MockEmbedding()
    index = GPTKnowledgeGraphIndex.from_documents(
        documents, include_embeddings=True, service_context=mock_service_context
    )
    retriever = KGTableRetriever(index, similarity_top_k=2)

    # returns only two rel texts to use for generating response
    # uses hyrbid query by default
    nodes = retriever.retrieve(QueryBundle("foo"))
    assert len(nodes) == 2
