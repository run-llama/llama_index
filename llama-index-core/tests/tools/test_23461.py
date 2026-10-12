import asyncio

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools.calling import call_tool, acall_tool


class FailingEngine(CustomQueryEngine):
    calls: int = 0

    def custom_query(self, query_str: str) -> Response:
        self.calls += 1
        raise RuntimeError("upstream 429")


def test_sync_tool_runs_once():
    engine = FailingEngine()
    tool = QueryEngineTool.from_defaults(
        query_engine=engine,
        name="docs",
        description="Search documents",
    )

    result = call_tool(tool, {"input": "refund policy"})

    assert engine.calls == 1, f"Expected 1 execution, got {engine.calls}"
    assert result.is_error is True
    assert isinstance(result.exception, RuntimeError)


def test_async_tool_runs_once():
    engine = FailingEngine()
    tool = QueryEngineTool.from_defaults(
        query_engine=engine,
        name="docs",
        description="Search documents",
    )

    result = asyncio.run(acall_tool(tool, {"input": "refund policy"}))

    assert engine.calls == 1, f"Expected 1 execution, got {engine.calls}"
    assert result.is_error is True
    assert isinstance(result.exception, RuntimeError)


from llama_index.core.tools import FunctionTool


def test_successful_tool_call():
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    tool = FunctionTool.from_defaults(fn=greet)

    result = call_tool(tool, {"name": "Alice"})

    assert result.is_error is False
    assert "Hello, Alice!" in result.content


def test_successful_async_tool_call():
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    tool = FunctionTool.from_defaults(async_fn=greet)

    result = asyncio.run(acall_tool(tool, {"name": "Alice"}))

    assert result.is_error is False
    assert "Hello, Alice!" in result.content
