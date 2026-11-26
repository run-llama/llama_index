"""Test document summary retrievers."""

import asyncio
from typing import Any, List

import pytest
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.indices.document_summary.base import (
    DocumentSummaryIndex,
    DocumentSummaryRetrieverMode,
)
from llama_index.core.indices.document_summary.retrievers import (
    DocumentSummaryIndexEmbeddingRetriever,
    DocumentSummaryIndexLLMRetriever,
)
from llama_index.core.llms import MockLLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.schema import BaseNode


def test_embedding_retriever(index: DocumentSummaryIndex) -> None:
    retriever = index.as_retriever()
    assert isinstance(retriever, DocumentSummaryIndexEmbeddingRetriever)
    results = retriever.retrieve("Test query")
    assert len(results) == 1
    assert results[0].node.ref_doc_id == "doc_4"

    retriever = index.as_retriever(similarity_top_k=2)
    assert isinstance(retriever, DocumentSummaryIndexEmbeddingRetriever)
    results = retriever.retrieve("Test query")
    assert len(results) == 2
    assert results[0].node.ref_doc_id == "doc_3"
    assert results[1].node.ref_doc_id == "doc_4"


def test_llm_retriever(
    index: DocumentSummaryIndex,
) -> None:
    retriever = index.as_retriever(retriever_mode=DocumentSummaryRetrieverMode.LLM)
    assert isinstance(retriever, DocumentSummaryIndexLLMRetriever)
    results = retriever.retrieve("Test query")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_llm_aretrieve_matches_retrieve_without_blocking(
    index: DocumentSummaryIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = ["Doc: 1, Relevance: 8\nDoc: 2, Relevance: 3", "Doc: 1, Relevance: 10"]
    pending = iter(answers)
    calls = []

    class RecordingLLM(MockLLM):
        def predict(self, prompt: BasePromptTemplate, **kwargs: Any) -> str:
            calls.append(("predict", kwargs))
            return next(pending)

        async def apredict(self, prompt: BasePromptTemplate, **kwargs: Any) -> str:
            calls.append(("apredict", kwargs))
            await asyncio.sleep(0)
            return next(pending)

    debug_handler = LlamaDebugHandler(print_trace_on_end=False)
    retriever = DocumentSummaryIndexLLMRetriever(
        index,
        llm=RecordingLLM(),
        choice_batch_size=2,
        choice_top_k=3,
        callback_manager=CallbackManager([debug_handler]),
    )
    expected = retriever.retrieve("Test query")
    sync_kwargs = [kwargs for _, kwargs in calls]
    assert [method for method, _ in calls] == ["predict", "predict"]
    pending = iter(answers)
    calls.clear()

    async_reads = []
    aget_nodes = index.docstore.aget_nodes

    async def read_nodes(
        node_ids: List[str], raise_error: bool = True
    ) -> List[BaseNode]:
        async_reads.append(node_ids)
        await asyncio.sleep(0)
        return await aget_nodes(node_ids, raise_error=raise_error)

    def reject_sync_read(*args: Any, **kwargs: Any) -> List[BaseNode]:
        raise AssertionError("Async retrieval must not use synchronous docstore reads")

    monkeypatch.setattr(index.docstore, "get_nodes", reject_sync_read)
    monkeypatch.setattr(index.docstore, "aget_nodes", read_nodes)

    progress = []

    async def heartbeat() -> None:
        progress.append("running")

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        actual = await retriever.aretrieve("Test query")
        assert progress == ["running"]
    finally:
        await heartbeat_task

    assert [method for method, _ in calls] == ["apredict", "apredict"]
    assert [kwargs for _, kwargs in calls] == sync_kwargs
    assert [(node.node_id, node.score) for node in actual] == [
        (node.node_id, node.score) for node in expected
    ]
    assert [node.node.ref_doc_id for node in actual] == ["doc_3", "doc_1", "doc_2"]
    assert [node.score for node in actual] == [10.0, 8.0, 3.0]
    summary_ids = index.index_struct.summary_ids
    assert async_reads == [
        summary_ids[:2],
        summary_ids[2:],
        index.index_struct.summary_id_to_node_ids[summary_ids[2]],
        index.index_struct.summary_id_to_node_ids[summary_ids[0]],
        index.index_struct.summary_id_to_node_ids[summary_ids[1]],
    ]
    assert len(debug_handler.get_event_pairs(CBEventType.RETRIEVE)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_index", [False, True])
async def test_llm_aretrieve_empty_results(
    index: DocumentSummaryIndex, empty_index: bool
) -> None:
    calls = []

    class EmptyLLM(MockLLM):
        def predict(self, prompt: BasePromptTemplate, **kwargs: Any) -> str:
            raise AssertionError("Async retrieval must not use predict")

        async def apredict(self, prompt: BasePromptTemplate, **kwargs: Any) -> str:
            calls.append(kwargs)
            return "No relevant documents."

    if empty_index:
        index.index_struct.summary_id_to_node_ids.clear()
    retriever = DocumentSummaryIndexLLMRetriever(index, llm=EmptyLLM())
    assert await retriever.aretrieve("Test query") == []
    assert len(calls) == (0 if empty_index else 1)


@pytest.mark.asyncio
async def test_llm_aretrieve_custom_selection(index: DocumentSummaryIndex) -> None:
    formatted = []
    parsed = []

    def format_batch(nodes: List[BaseNode]) -> str:
        formatted.append([node.node_id for node in nodes])
        return "custom context"

    def parse_answer(answer: str, num_choices: int) -> tuple[list[int], list[float]]:
        parsed.append((answer, num_choices))
        return [num_choices], [0.0]

    class CustomLLM(MockLLM):
        async def apredict(self, prompt: BasePromptTemplate, **kwargs: Any) -> str:
            assert kwargs["context_str"] == "custom context"
            assert kwargs["query_str"] == "Test query"
            return "custom answer"

    retriever = DocumentSummaryIndexLLMRetriever(
        index,
        llm=CustomLLM(),
        choice_batch_size=3,
        choice_top_k=2,
        format_node_batch_fn=format_batch,
        parse_choice_select_answer_fn=parse_answer,
    )
    results = await retriever.aretrieve("Test query")
    summary_ids = index.index_struct.summary_ids
    assert formatted == [summary_ids[:3], summary_ids[3:]]
    assert parsed == [("custom answer", 3), ("custom answer", 1)]
    assert [node.node.ref_doc_id for node in results] == ["doc_3", "doc_4"]
    assert [node.score for node in results] == [0.0, 0.0]


@pytest.mark.asyncio
async def test_llm_aretrieve_propagates_errors(index: DocumentSummaryIndex) -> None:
    class FailingLLM(MockLLM):
        async def apredict(self, prompt: BasePromptTemplate, **kwargs: Any) -> str:
            raise ValueError("model failed")

    retriever = DocumentSummaryIndexLLMRetriever(index, llm=FailingLLM())
    with pytest.raises(ValueError, match="model failed"):
        await retriever.aretrieve("Test query")


@pytest.mark.asyncio
async def test_llm_aretrieve_cancellation_stops_batches(
    index: DocumentSummaryIndex,
) -> None:
    started = asyncio.Event()
    calls = []

    class WaitingLLM(MockLLM):
        async def apredict(self, prompt: BasePromptTemplate, **kwargs: Any) -> str:
            calls.append(kwargs)
            started.set()
            await asyncio.Event().wait()
            return "Doc: 1, Relevance: 8"

    retriever = DocumentSummaryIndexLLMRetriever(
        index, llm=WaitingLLM(), choice_batch_size=1
    )
    task = asyncio.create_task(retriever.aretrieve("Test query"))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
    finally:
        task.cancel()
        outcome = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(outcome[0], asyncio.CancelledError)
    assert len(calls) == 1
