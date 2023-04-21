from typing import Any, List
from unittest.mock import patch
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from gpt_index.indices.knowledge_graph.retrievers import KGTableRetriever
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.readers.schema.base import Document
from tests.indices.knowledge_graph.test_base import (
    mock_extract_triplets,
    mock_get_text_embedding,
    mock_get_text_embeddings,
)
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_prompts import MOCK_QUERY_KEYWORD_EXTRACT_PROMPT


@patch_common
@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_as_retriever(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Any,
    documents: List[Document],
) -> None:
    """Test query."""
    index = GPTKnowledgeGraphIndex.from_documents(documents)
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


@patch_common
@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retrievers(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Any,
    documents: List[Document],
) -> None:
    # test specific retriever class
    index = GPTKnowledgeGraphIndex.from_documents(documents)
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


@patch_common
@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retriever_no_text(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Any,
    documents: List[Document],
) -> None:
    # test specific retriever class
    index = GPTKnowledgeGraphIndex.from_documents(documents)
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


@patch_common
@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_retrieve_similarity(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    _mock_get_text_embeddings: Any,
    _mock_get_text_embedding: Any,
    struct_kwargs: Any,
    documents: List[Document],
) -> None:
    """Test query."""
    index = GPTKnowledgeGraphIndex.from_documents(documents, include_embeddings=True)
    retriever = KGTableRetriever(index, similarity_top_k=2)

    # returns only two rel texts to use for generating response
    # uses hyrbid query by default
    nodes = retriever.retrieve(QueryBundle("foo"))
    assert len(nodes) == 2
