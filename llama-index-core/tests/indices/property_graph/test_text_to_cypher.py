from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher import (
    DEFAULT_SUMMARY_TEMPLATE,
    TextToCypherRetriever,
)
from llama_index.core.llms import MockLLM
from llama_index.core.prompts import PromptTemplate


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("summarize", [False, True])
@pytest.mark.parametrize("custom_template", [False, True])
async def test_summary_preserves_user_question(
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
    summarize: bool,
    custom_template: bool,
) -> None:
    question = "How many flights are there? Answer in Spanish."
    cypher = "MATCH (f:Flight) RETURN count(f) AS count"
    rows = [{"count": 42, "internal": "hidden"}]
    cleaned_rows = [{"count": 42}]
    graph_store = MagicMock(spec=PropertyGraphStore)
    graph_store.supports_structured_queries = True
    graph_store.get_schema_str.return_value = "Flight"
    graph_store.aget_schema_str = AsyncMock(return_value="Flight")
    graph_store.structured_query.return_value = rows
    graph_store.astructured_query = AsyncMock(return_value=rows)
    complete = MagicMock(
        side_effect=[
            CompletionResponse(text=cypher),
            CompletionResponse(text="42 vuelos"),
        ]
    )
    monkeypatch.setattr(MockLLM, "complete", complete)
    summary_template = (
        PromptTemplate("Question: {question}\nResults: {context}")
        if custom_template
        else DEFAULT_SUMMARY_TEMPLATE
    )
    retriever = TextToCypherRetriever(
        graph_store,
        llm=MockLLM(),
        text_to_cypher_template="Schema: {schema}\nQuestion: {question}",
        summarize_response=summarize,
        summarization_template=summary_template if custom_template else None,
        allowed_output_fields=["count"],
        include_raw_response_as_metadata=True,
    )

    nodes = (
        await retriever.aretrieve(question)
        if use_async
        else retriever.retrieve(question)
    )

    assert len(nodes) == 1
    assert nodes[0].node.metadata == {"query": cypher, "response": cleaned_rows}
    if use_async:
        graph_store.astructured_query.assert_awaited_once_with(cypher)
        graph_store.structured_query.assert_not_called()
    else:
        graph_store.structured_query.assert_called_once_with(cypher)
        graph_store.astructured_query.assert_not_called()
    if summarize:
        assert complete.call_count == 2
        assert complete.call_args.args[0] == summary_template.format(
            question=question, context=str(cleaned_rows)
        )
        assert nodes[0].node.text == "42 vuelos"
    else:
        assert complete.call_count == 1
        assert nodes[0].node.text == retriever.response_template.format(
            query=cypher, response=str(cleaned_rows)
        )
