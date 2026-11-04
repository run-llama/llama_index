from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher import (
    TextToCypherRetriever,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

QUESTION = "How many flights are there? Answer in Spanish."
CYPHER = "MATCH (f:Flight) RETURN count(f) AS count"
SUMMARY = "Hay 42 vuelos."
GRAPH_ROWS = [{"count": 42}]
CUSTOM_SUMMARY_TEMPLATE = PromptTemplate(
    "Summarize for the user.\nQuestion: {question}\nInformation: {context}\nHelpful Answer:"
)


class _ScriptedLLM:
    """Return generated Cypher, then a summary, recording each formatted prompt."""

    def __init__(self) -> None:
        self.formatted_prompts: List[str] = []
        self.prompt_args: List[dict] = []

    def predict(self, prompt: PromptTemplate, **prompt_args: Any) -> str:
        self.formatted_prompts.append(prompt.format(**prompt_args))
        self.prompt_args.append(prompt_args)
        if len(self.formatted_prompts) == 1:
            return CYPHER
        return SUMMARY

    async def apredict(self, prompt: PromptTemplate, **prompt_args: Any) -> str:
        return self.predict(prompt, **prompt_args)


def _graph_store() -> MagicMock:
    store = MagicMock()
    store.supports_structured_queries = True
    store.get_schema_str.return_value = "(:Flight)"
    store.aget_schema_str = AsyncMock(return_value="(:Flight)")
    store.structured_query.return_value = GRAPH_ROWS
    store.astructured_query = AsyncMock(return_value=GRAPH_ROWS)
    return store


def _retriever(
    llm: _ScriptedLLM,
    graph_store: MagicMock,
    summarization_template: Optional[PromptTemplate],
) -> TextToCypherRetriever:
    return TextToCypherRetriever(
        graph_store=graph_store,
        llm=llm,  # type: ignore[arg-type]
        text_to_cypher_template="Schema:\n{schema}\nQuestion:\n{question}\nCypher:",
        summarize_response=True,
        summarization_template=summarization_template,
        include_raw_response_as_metadata=True,
    )


def _assert_summary_uses_original_question(
    llm: _ScriptedLLM, nodes: List[NodeWithScore]
) -> None:
    assert len(llm.formatted_prompts) == 2
    summary_prompt = llm.formatted_prompts[1]
    assert QUESTION in summary_prompt
    assert CYPHER not in summary_prompt
    assert llm.prompt_args[1]["question"] == QUESTION
    node = nodes[0].node
    assert isinstance(node, TextNode)
    assert node.text == SUMMARY
    assert nodes[0].node.metadata["query"] == CYPHER
    assert nodes[0].node.metadata["response"] == GRAPH_ROWS


@pytest.mark.parametrize(
    "summarization_template",
    [None, CUSTOM_SUMMARY_TEMPLATE],
    ids=["default_template", "custom_template"],
)
def test_retrieve_from_graph_summarizes_original_question(
    summarization_template: Optional[PromptTemplate],
) -> None:
    llm = _ScriptedLLM()
    graph_store = _graph_store()
    retriever = _retriever(llm, graph_store, summarization_template)

    nodes = retriever.retrieve_from_graph(QueryBundle(query_str=QUESTION))

    _assert_summary_uses_original_question(llm, nodes)
    graph_store.structured_query.assert_called_once_with(CYPHER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "summarization_template",
    [None, CUSTOM_SUMMARY_TEMPLATE],
    ids=["default_template", "custom_template"],
)
async def test_aretrieve_from_graph_summarizes_original_question(
    summarization_template: Optional[PromptTemplate],
) -> None:
    llm = _ScriptedLLM()
    graph_store = _graph_store()
    retriever = _retriever(llm, graph_store, summarization_template)

    nodes = await retriever.aretrieve_from_graph(QueryBundle(query_str=QUESTION))

    _assert_summary_uses_original_question(llm, nodes)
    graph_store.astructured_query.assert_awaited_once_with(CYPHER)
