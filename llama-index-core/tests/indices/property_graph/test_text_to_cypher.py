from typing import Any, Dict, List, Optional

import pytest
from pydantic import PrivateAttr

from llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher import (
    TextToCypherRetriever,
)
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import QueryBundle

USER_QUESTION = "How many people live in Paris? Answer in Spanish."
GENERATED_CYPHER = "MATCH (c:City {name: 'Paris'}) RETURN c.population"

CYPHER_TEMPLATE = PromptTemplate("Turn this into Cypher: {question}\nSchema: {schema}")


class RecordingMockLLM(MockLLM):
    """MockLLM that records predict/apredict calls and returns canned Cypher."""

    _predict_calls: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def predict_calls(self) -> List[Dict[str, Any]]:
        return self._predict_calls

    def predict(self, prompt: Any, **prompt_args: Any) -> str:
        self._predict_calls.append({"prompt": prompt, "args": prompt_args})
        if prompt is CYPHER_TEMPLATE:
            return GENERATED_CYPHER
        return super().predict(prompt, **prompt_args)

    async def apredict(self, prompt: Any, **prompt_args: Any) -> str:
        self._predict_calls.append({"prompt": prompt, "args": prompt_args})
        if prompt is CYPHER_TEMPLATE:
            return GENERATED_CYPHER
        return await super().apredict(prompt, **prompt_args)


class FakePropertyGraphStore:
    """Minimal stand-in for a PropertyGraphStore supporting Cypher."""

    supports_structured_queries = True
    text_to_cypher_template = CYPHER_TEMPLATE

    def get_schema_str(self) -> str:
        return "Node: City(name, population)"

    def structured_query(self, query: str) -> List[Dict[str, Any]]:
        assert query == GENERATED_CYPHER
        return [{"name": "Paris", "population": 2200000}]

    async def aget_schema_str(self) -> str:
        return self.get_schema_str()

    async def astructured_query(self, query: str) -> List[Dict[str, Any]]:
        return self.structured_query(query)


def _make_retriever(llm: RecordingMockLLM) -> TextToCypherRetriever:
    return TextToCypherRetriever(
        graph_store=FakePropertyGraphStore(),
        llm=llm,
        text_to_cypher_template=CYPHER_TEMPLATE,
        summarize_response=True,
    )


def test_summarizer_receives_user_question_not_cypher() -> None:
    llm = RecordingMockLLM()
    retriever = _make_retriever(llm)

    nodes = retriever.retrieve_from_graph(QueryBundle(USER_QUESTION))

    assert len(nodes) == 1
    assert len(llm.predict_calls) == 2

    summary_kwargs = llm.predict_calls[1]["args"]
    assert summary_kwargs["question"] == USER_QUESTION
    assert GENERATED_CYPHER not in summary_kwargs["question"]
    assert "population" in summary_kwargs["context"]


@pytest.mark.asyncio
async def test_async_summarizer_receives_user_question_not_cypher() -> None:
    llm = RecordingMockLLM()
    retriever = _make_retriever(llm)

    nodes = await retriever.aretrieve_from_graph(QueryBundle(USER_QUESTION))

    assert len(nodes) == 1
    assert len(llm.predict_calls) == 2

    summary_kwargs = llm.predict_calls[1]["args"]
    assert summary_kwargs["question"] == USER_QUESTION
    assert GENERATED_CYPHER not in summary_kwargs["question"]


def test_text_to_cypher_step_still_gets_user_question() -> None:
    llm = RecordingMockLLM()
    retriever = _make_retriever(llm)

    retriever.retrieve_from_graph(QueryBundle(USER_QUESTION))

    cypher_kwargs = llm.predict_calls[0]["args"]
    assert cypher_kwargs["question"] == USER_QUESTION
