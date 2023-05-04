from typing import Any, List
from unittest.mock import patch
from llama_index.indices.list.base import GPTListIndex
from llama_index.indices.list.retrievers import ListIndexEmbeddingRetriever
from llama_index.indices.service_context import ServiceContext
from llama_index.readers.schema.base import Document
from tests.indices.list.test_index import _get_embeddings


def test_retrieve_default(
    documents: List[Document], mock_service_context: ServiceContext
) -> None:
    """Test list query."""
    index = GPTListIndex.from_documents(documents, service_context=mock_service_context)

    query_str = "What is?"
    retriever = index.as_retriever(retriever_mode="default")
    nodes = retriever.retrieve(query_str)

    for node_with_score, line in zip(nodes, documents[0].get_text().split("\n")):
        assert node_with_score.node.text == line


@patch.object(
    ListIndexEmbeddingRetriever,
    "_get_embeddings",
    side_effect=_get_embeddings,
)
def test_embedding_query(
    _patch_get_embeddings: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test embedding query."""
    index = GPTListIndex.from_documents(documents, service_context=mock_service_context)

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(retriever_mode="embedding", similarity_top_k=1)
    nodes = retriever.retrieve(query_str)
    assert len(nodes) == 1

    assert nodes[0].node.text == "Hello world."
