from typing import Any, List
from unittest.mock import patch

from llama_index.indices.list.base import ListIndex
from llama_index.indices.list.retrievers import ListIndexEmbeddingRetriever
from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.choice_select import ChoiceSelectPrompt
from llama_index.prompts.prompts import Prompt
from llama_index.schema import Document
from tests.indices.list.test_index import _get_embeddings


def test_retrieve_default(
    documents: List[Document], mock_service_context: ServiceContext
) -> None:
    """Test list query."""
    index = ListIndex.from_documents(documents, service_context=mock_service_context)

    query_str = "What is?"
    retriever = index.as_retriever(retriever_mode="default")
    nodes = retriever.retrieve(query_str)

    for node_with_score, line in zip(nodes, documents[0].get_content().split("\n")):
        assert node_with_score.node.get_content() == line


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
    index = ListIndex.from_documents(documents, service_context=mock_service_context)

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(retriever_mode="embedding", similarity_top_k=1)
    nodes = retriever.retrieve(query_str)
    assert len(nodes) == 1

    assert nodes[0].node.get_content() == "Hello world."


def mock_llmpredictor_predict(self: Any, prompt: Prompt, **prompt_args: Any) -> str:
    """Patch llm predictor predict."""
    assert isinstance(prompt, ChoiceSelectPrompt)
    return "Doc: 2, Relevance: 5"


@patch.object(
    LLMPredictor,
    "predict",
    mock_llmpredictor_predict,
)
def test_llm_query(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test llm query."""
    index = ListIndex.from_documents(documents, service_context=mock_service_context)

    # test llm query (batch size 10)
    query_str = "What is?"
    retriever = index.as_retriever(retriever_mode="llm")
    nodes = retriever.retrieve(query_str)
    assert len(nodes) == 1

    assert nodes[0].node.get_content() == "This is a test."

    # test llm query (batch size 2)
    query_str = "What is?"
    retriever = index.as_retriever(retriever_mode="llm", choice_batch_size=2)
    nodes = retriever.retrieve(query_str)
    assert len(nodes) == 2

    assert nodes[0].node.get_content() == "This is a test."
    assert nodes[1].node.get_content() == "This is a test v2."
