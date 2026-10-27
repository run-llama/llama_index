from typing import Any, Dict, List, Optional, Sequence

import pytest

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.graph_stores.types import (
    LabelledNode,
    PropertyGraphStore,
    Relation,
    Triplet,
)
from llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher import (
    TextToCypherRetriever,
)
from llama_index.core.llms import MockLLM
from llama_index.core.schema import QueryBundle

QUESTION = "How many flights are there? Answer in Spanish."
CYPHER = "MATCH (f:Flight) RETURN count(f) AS count"


class _FakeGraphStore(PropertyGraphStore):
    supports_structured_queries: bool = True

    def get_schema_str(self, refresh: bool = False) -> str:
        return "(:Flight)"

    async def aget_schema_str(self, refresh: bool = False) -> str:
        return "(:Flight)"

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        return [{"count": 42}]

    async def astructured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        return [{"count": 42}]

    def get(
        self, properties: Optional[dict] = None, ids: Optional[List[str]] = None
    ) -> List[LabelledNode]:
        return []

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        return []

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        return []

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        pass

    def upsert_relations(self, relations: List[Relation]) -> None:
        pass

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        pass

    def vector_query(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _RecordingLLM(MockLLM):
    """MockLLM that records prompts and replays scripted completions."""

    prompts: List[str] = []
    replies: List[str] = []

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.prompts.append(prompt)
        index = len(self.prompts) - 1
        text = self.replies[index] if index < len(self.replies) else ""
        return CompletionResponse(text=text)


def _build_retriever(
    summarize_response: bool = True,
    summarization_template: Optional[str] = None,
) -> tuple[TextToCypherRetriever, _RecordingLLM]:
    llm = _RecordingLLM()
    llm.prompts = []
    llm.replies = [CYPHER, "Hay 42 vuelos."]
    retriever = TextToCypherRetriever(
        graph_store=_FakeGraphStore(),
        llm=llm,
        summarize_response=summarize_response,
        summarization_template=summarization_template,
    )
    return retriever, llm


def test_summarization_prompt_uses_original_question() -> None:
    retriever, llm = _build_retriever()

    retriever.retrieve_from_graph(QueryBundle(query_str=QUESTION))

    assert len(llm.prompts) == 2
    summarization_prompt = llm.prompts[1]
    assert QUESTION in summarization_prompt
    assert f"Information:\n        [{{'count': 42}}]" in summarization_prompt


def test_custom_summarization_prompt_uses_original_question() -> None:
    retriever, llm = _build_retriever(
        summarization_template="Q: {question}\nInfo: {context}\nA:"
    )

    retriever.retrieve_from_graph(QueryBundle(query_str=QUESTION))

    assert llm.prompts[1].startswith(f"Q: {QUESTION}")
    assert CYPHER not in llm.prompts[1]


@pytest.mark.asyncio
async def test_async_summarization_prompt_uses_original_question() -> None:
    retriever, llm = _build_retriever()

    await retriever.aretrieve_from_graph(QueryBundle(query_str=QUESTION))

    assert QUESTION in llm.prompts[1]
    assert f"Question:\n        {CYPHER}" not in llm.prompts[1]


def test_raw_response_metadata_still_keeps_cypher() -> None:
    retriever, _ = _build_retriever()
    retriever.include_raw_response_as_metadata = True

    nodes = retriever.retrieve_from_graph(QueryBundle(query_str=QUESTION))

    assert nodes[0].node.metadata["query"] == CYPHER
    assert nodes[0].node.metadata["response"] == [{"count": 42}]
