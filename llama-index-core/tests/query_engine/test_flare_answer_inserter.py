"""Tests for DirectLookaheadAnswerInserter offset handling."""

from typing import Any

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.llms import CompletionResponse, LLMMetadata
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.query_engine.flare.answer_inserter import (
    DirectLookaheadAnswerInserter,
)
from llama_index.core.query_engine.flare.base import FLAREInstructQueryEngine
from llama_index.core.query_engine.flare.output_parser import QueryTaskOutputParser

LOOKAHEAD = (
    "Red is for [Search(why is red on the flag?)], green for "
    "[Search(why is green on the flag?)], and gold for mineral wealth."
)
ANSWERS = {
    "why is red on the flag?": "the blood of those who died for independence",
    "why is green on the flag?": "the forests and farms",
}
EXPECTED = (
    "Red is for the blood of those who died for independence, green for "
    "the forests and farms, and gold for mineral wealth."
)


def test_insert_replaces_multiple_tags_at_their_offsets() -> None:
    """Every tag is replaced by its own answer, in place."""
    tasks = QueryTaskOutputParser().parse(LOOKAHEAD)
    answers = [ANSWERS[t.query_str] for t in tasks]

    out = DirectLookaheadAnswerInserter().insert(LOOKAHEAD, tasks, answers)

    assert out == EXPECTED


def test_insert_single_tag() -> None:
    """The default one-task-per-pass path is unchanged."""
    response = "Red is for [Search(why is red on the flag?)], and so on."
    tasks = QueryTaskOutputParser().parse(response)
    answers = [ANSWERS[tasks[0].query_str]]

    out = DirectLookaheadAnswerInserter().insert(response, tasks, answers)

    assert out == "Red is for the blood of those who died for independence, and so on."


class _FlareLLM(CustomLLM):
    calls: int = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=4096, num_output=256, is_chat_model=False)

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.calls += 1
        return CompletionResponse(text=LOOKAHEAD if self.calls == 1 else "done")

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Any:
        raise NotImplementedError


class _StubQueryEngine(BaseQueryEngine):
    def __init__(self) -> None:
        super().__init__(callback_manager=CallbackManager([]))

    def _get_prompt_modules(self) -> dict:
        return {}

    def _query(self, query_bundle: Any) -> Response:
        return Response(response=ANSWERS[query_bundle.query_str], source_nodes=[])

    async def _aquery(self, query_bundle: Any) -> Response:
        raise NotImplementedError


def test_flare_engine_with_direct_inserter_and_multiple_tasks() -> None:
    """End to end: two query tasks in one lookahead, direct inserter."""
    engine = FLAREInstructQueryEngine(
        query_engine=_StubQueryEngine(),
        llm=_FlareLLM(),
        lookahead_answer_inserter=DirectLookaheadAnswerInserter(),
        max_lookahead_query_tasks=2,
        verbose=False,
    )

    result = engine.query("What do the colors on Ghana's flag mean?")

    assert str(result).strip() == EXPECTED
