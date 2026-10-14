from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest
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


def _make_recording_llm(
    answers: List[str], calls: List[str], allow_sync: bool = True
) -> MockLLM:
    """Build a MockLLM that replays `answers` in order and records each call."""
    sync_answers = iter(answers)
    async_answers = iter(answers)

    class RecordingLLM(MockLLM):
        def predict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
            if not allow_sync:
                raise AssertionError(
                    "sync predict() must not be called from _aretrieve"
                )
            calls.append("predict")
            return next(sync_answers)

        async def apredict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
            calls.append("apredict")
            return next(async_answers)

    return RecordingLLM()


@pytest.mark.asyncio
async def test_llm_aretrieve_uses_async_predict(
    documents: List[Document], patch_token_text_splitter
) -> None:
    """Test that aretrieve awaits llm.apredict instead of blocking on predict."""
    index = SummaryIndex.from_documents(documents)

    calls: List[str] = []
    llm = _make_recording_llm(["Doc: 2, Relevance: 5"], calls, allow_sync=False)
    retriever = index.as_retriever(retriever_mode="llm", llm=llm)
    nodes = await retriever.aretrieve("What is?")

    assert calls == ["apredict"]
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "This is a test."


@pytest.mark.asyncio
async def test_llm_aretrieve_matches_retrieve(
    documents: List[Document], patch_token_text_splitter
) -> None:
    """Test that aretrieve and retrieve agree across multiple batches."""
    index = SummaryIndex.from_documents(documents)
    # one answer per batch (batch size 2 over 4 nodes)
    answers = ["Doc: 2, Relevance: 5", "Doc: 1, Relevance: 8"]

    sync_calls: List[str] = []
    sync_retriever = index.as_retriever(
        retriever_mode="llm",
        llm=_make_recording_llm(answers, sync_calls),
        choice_batch_size=2,
    )
    sync_nodes = sync_retriever.retrieve("What is?")

    async_calls: List[str] = []
    async_retriever = index.as_retriever(
        retriever_mode="llm",
        llm=_make_recording_llm(answers, async_calls),
        choice_batch_size=2,
    )
    async_nodes = await async_retriever.aretrieve("What is?")

    assert sync_calls == ["predict", "predict"]
    assert async_calls == ["apredict", "apredict"]
    assert len(sync_nodes) == 2
    assert len(async_nodes) == len(sync_nodes)
    for sync_node, async_node in zip(sync_nodes, async_nodes):
        assert async_node.node.node_id == sync_node.node.node_id
        assert async_node.score == sync_node.score
