from typing import Any, Dict, List, Tuple
from unittest.mock import patch

from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.indices.list.retrievers import SummaryIndexEmbeddingRetriever
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.schema import BaseNode, Document


def _get_embeddings(
    query_str: str, nodes: List[BaseNode]
) -> Tuple[List[float], List[List[float]]]:
    """Get node text embedding similarity."""
    text_embed_map: Dict[str, List[float]] = {
        "Hello world.": [1.0, 0.0, 0.0, 0.0, 0.0],
        "This is a test.": [0.0, 1.0, 0.0, 0.0, 0.0],
        "This is another test.": [0.0, 0.0, 1.0, 0.0, 0.0],
        "This is a test v2.": [0.0, 0.0, 0.0, 1.0, 0.0],
    }
    node_embeddings = []
    for node in nodes:
        node_embeddings.append(text_embed_map[node.get_content()])

    return [1.0, 0, 0, 0, 0], node_embeddings


def test_retrieve_default(documents: List[Document], patch_token_text_splitter) -> None:
    """Test list query."""
    index = SummaryIndex.from_documents(documents)

    query_str = "What is?"
    retriever = index.as_retriever(retriever_mode="default")
    nodes = retriever.retrieve(query_str)

    for node_with_score, line in zip(nodes, documents[0].get_content().split("\n")):
        assert node_with_score.node.get_content() == line


@patch.object(
    SummaryIndexEmbeddingRetriever,
    "_get_embeddings",
    side_effect=_get_embeddings,
)
def test_embedding_query(
    _patch_get_embeddings: Any, documents: List[Document], patch_token_text_splitter
) -> None:
    """Test embedding query."""
    index = SummaryIndex.from_documents(documents)

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(retriever_mode="embedding", similarity_top_k=1)
    nodes = retriever.retrieve(query_str)
    assert len(nodes) == 1

    assert nodes[0].node.get_content() == "Hello world."


def mock_llmpredictor_predict(
    self: Any, prompt: BasePromptTemplate, **prompt_args: Any
) -> str:
    """Patch llm predictor predict."""
    return "Doc: 2, Relevance: 5"


@patch.object(
    MockLLM,
    "predict",
    mock_llmpredictor_predict,
)
def test_llm_query(documents: List[Document], patch_token_text_splitter) -> None:
    """Test llm query."""
    index = SummaryIndex.from_documents(documents)

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
