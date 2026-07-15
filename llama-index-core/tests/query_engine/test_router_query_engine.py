import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.base.base_selector import (
    BaseSelector,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.base.response.schema import Response
from llama_index.core.llms.mock import MockLLM
from llama_index.core.query_engine.router_query_engine import (
    RouterQueryEngine,
    ToolRetrieverRouterQueryEngine,
)
from llama_index.core.tools.types import ToolMetadata


class _AlwaysMultiSelector(BaseSelector):
    def _get_prompts(self):
        return {}

    def _update_prompts(self, prompts):
        pass

    def _select(self, choices, query):
        return SelectorResult(
            selections=[
                SingleSelection(index=i, reason="") for i in range(len(choices))
            ]
        )

    async def _aselect(self, choices, query):
        return self._select(choices, query)


def _make_query_engine_tool(name: str):
    async def _fake_aquery(_):
        await asyncio.sleep(0.05)
        return Response(response="ok")

    engine = MagicMock()
    engine.aquery = AsyncMock(side_effect=_fake_aquery)

    tool = MagicMock()
    tool.query_engine = engine
    tool.metadata = ToolMetadata(name=name, description=name)
    return tool


class _MockSummarizer:
    async def aget_response(self, *args, **kwargs):
        return "combined"


async def _assert_not_blocked(coro) -> None:
    loop = asyncio.get_running_loop()
    start = loop.time()
    ran_at = None

    async def _background():
        nonlocal ran_at
        await asyncio.sleep(0.01)
        ran_at = loop.time()

    bg_task = asyncio.create_task(_background())
    await asyncio.sleep(0)
    await coro
    await bg_task

    assert ran_at is not None, "background task never ran"
    assert (ran_at - start) < 0.04, (
        f"background task finished {ran_at - start:.3f}s after start "
        f"(expected < 0.04s) — the event loop was likely blocked"
    )


@pytest.mark.asyncio
async def test_router_aquery_does_not_block_event_loop():
    tool_a = _make_query_engine_tool("a")
    tool_b = _make_query_engine_tool("b")

    router = RouterQueryEngine(
        selector=_AlwaysMultiSelector(),
        query_engine_tools=[tool_a, tool_b],
        llm=MockLLM(),
        summarizer=_MockSummarizer(),
    )

    await _assert_not_blocked(router.aquery("test query"))

    assert tool_a.query_engine.aquery.call_count == 1
    assert tool_b.query_engine.aquery.call_count == 1


@pytest.mark.asyncio
async def test_tool_retriever_router_aquery_does_not_block_event_loop():
    tool_a = _make_query_engine_tool("a")
    tool_b = _make_query_engine_tool("b")

    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=[tool_a, tool_b])

    router = ToolRetrieverRouterQueryEngine(
        retriever=retriever,
        llm=MockLLM(),
        summarizer=_MockSummarizer(),
    )

    await _assert_not_blocked(router.aquery("test query"))

    assert tool_a.query_engine.aquery.call_count == 1
    assert tool_b.query_engine.aquery.call_count == 1
