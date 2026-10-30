"""Regression tests for TextToCypherRetriever response summarization.

Bug (issue #23673): with summarize_response=True, both retrieve_from_graph()
and aretrieve_from_graph() passed the generated Cypher query in the `question`
slot of the summarization LLM call instead of the original natural-language
request. As a result, any instruction embedded in the original question (e.g.
"Answer in Spanish.") was lost during summarization.

These tests use a recording mock LLM that captures the prompt_args of every
predict/apredict call. The second call is the summarization call; we assert that
its `question` argument equals the original query string, not the generated Cypher.
"""

from typing import Any, List, Optional

from llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher import (
    TextToCypherRetriever,
)
from llama_index.core.llms import CustomLLM
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle


class _MockGraphStore:
    """Minimal duck-typed graph store: only the methods the retriever touches."""

    supports_structured_queries = True

    def get_schema_str(self, refresh: bool = False) -> str:
        return "MATCH (n) RETURN n"

    async def aget_schema_str(self, refresh: bool = False) -> str:
        return "MATCH (n) RETURN n"

    def structured_query(self, query: str, param_map: Optional[dict] = None) -> Any:
        return [{"count": 42}]

    async def astructured_query(self, query: str, param_map: Optional[dict] = None) -> Any:
        return [{"count": 42}]


class _RecordingLLM(CustomLLM):
    """Records prompt_args of every predict/apredict call.

    The first call is the text-to-Cypher generation (returns a Cypher string);
    subsequent calls are summarization (return canned text). We only care about
    what `question` value was passed to each call.
    """

    calls: List[dict] = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_chat_model=False)

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=prompt)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        def gen():
            yield CompletionResponse(text=prompt, delta=prompt)

        return gen()

    def predict(self, prompt: Any, **prompt_args: Any) -> str:
        self.calls.append(dict(prompt_args))
        if len(self.calls) == 1:
            return "MATCH (f:Flight) RETURN count(f) AS count"
        return "There are 42 flights."

    async def apredict(self, prompt: Any, **prompt_args: Any) -> str:
        self.calls.append(dict(prompt_args))
        if len(self.calls) == 1:
            return "MATCH (f:Flight) RETURN count(f) AS count"
        return "There are 42 flights."


ORIGINAL_QUESTION = "How many flights are there? Answer in Spanish."
_TEXT_TO_CYPHER_TEMPLATE = PromptTemplate(
    "Schema:\n{schema}\nQuestion: {question}\nCypher:"
)
_CUSTOM_SUMMARY_TEMPLATE = PromptTemplate(
    "Question: {question}\nInformation:\n{context}\nHelpful Answer:"
)


def _make_retriever(summarize_response: bool = True, custom_summary: bool = False):
    llm = _RecordingLLM()
    llm.calls = []
    retriever = TextToCypherRetriever(
        graph_store=_MockGraphStore(),
        llm=llm,
        text_to_cypher_template=_TEXT_TO_CYPHER_TEMPLATE,
        summarize_response=summarize_response,
        summarization_template=_CUSTOM_SUMMARY_TEMPLATE if custom_summary else None,
    )
    return retriever, llm


def test_sync_default_template_passes_original_question() -> None:
    """sync + default summary template: summarizer must see the original question."""
    retriever, llm = _make_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str=ORIGINAL_QUESTION))
    assert isinstance(nodes[0], NodeWithScore)
    # call 0 = text-to-cypher, call 1 = summarization
    assert len(llm.calls) == 2
    assert llm.calls[1]["question"] == ORIGINAL_QUESTION
    assert llm.calls[1]["question"] != "MATCH (f:Flight) RETURN count(f) AS count"


def test_async_default_template_passes_original_question() -> None:
    """async + default summary template: summarizer must see the original question."""
    import asyncio

    retriever, llm = _make_retriever()
    nodes = asyncio.run(retriever.aretrieve(QueryBundle(query_str=ORIGINAL_QUESTION)))
    assert isinstance(nodes[0], NodeWithScore)
    assert len(llm.calls) == 2
    assert llm.calls[1]["question"] == ORIGINAL_QUESTION


def test_sync_custom_template_passes_original_question() -> None:
    """sync + custom summary template: summarizer must see the original question."""
    retriever, llm = _make_retriever(custom_summary=True)
    retriever.retrieve(QueryBundle(query_str=ORIGINAL_QUESTION))
    assert len(llm.calls) == 2
    assert llm.calls[1]["question"] == ORIGINAL_QUESTION


def test_async_custom_template_passes_original_question() -> None:
    """async + custom summary template: summarizer must see the original question."""
    import asyncio

    retriever, llm = _make_retriever(custom_summary=True)
    asyncio.run(retriever.aretrieve(QueryBundle(query_str=ORIGINAL_QUESTION)))
    assert len(llm.calls) == 2
    assert llm.calls[1]["question"] == ORIGINAL_QUESTION


def test_summarize_response_false_makes_single_llm_call() -> None:
    """Control: without summarization there is only the text-to-Cypher call."""
    retriever, llm = _make_retriever(summarize_response=False)
    retriever.retrieve(QueryBundle(query_str=ORIGINAL_QUESTION))
    assert len(llm.calls) == 1
    assert llm.calls[0]["question"] == ORIGINAL_QUESTION
