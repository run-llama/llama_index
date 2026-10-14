"""Tests for tool invocation helpers."""

import pytest

from llama_index.core.tools.calling import acall_tool, call_tool
from llama_index.core.tools.function_tool import FunctionTool


def test_call_tool_does_not_retry_failed_function_tool() -> None:
    calls = []

    def remote_write(input: str) -> str:
        calls.append(input)
        raise RuntimeError("response lost after request was accepted")

    tool = FunctionTool.from_defaults(remote_write)

    output = call_tool(tool, {"input": "charge"})

    assert output.is_error
    assert calls == ["charge"]


@pytest.mark.asyncio
async def test_acall_tool_does_not_retry_failed_function_tool() -> None:
    calls = []

    async def remote_write(input: str) -> str:
        calls.append(input)
        raise RuntimeError("response lost after request was accepted")

    tool = FunctionTool.from_defaults(async_fn=remote_write)

    output = await acall_tool(tool, {"input": "charge"})

    assert output.is_error
    assert calls == ["charge"]


def test_call_tool_supports_keyword_only_function_tool() -> None:
    def lookup(*, query: str) -> str:
        return query

    tool = FunctionTool.from_defaults(lookup)

    output = call_tool(tool, {"query": "LlamaIndex"})

    assert output.raw_output == "LlamaIndex"


def test_call_tool_supports_positional_only_function_tool() -> None:
    def lookup(query: str, /) -> str:
        return query

    tool = FunctionTool.from_defaults(lookup)

    output = call_tool(tool, {"query": "LlamaIndex"})

    assert output.raw_output == "LlamaIndex"


def test_call_tool_does_not_retry_failed_query_engine_tool() -> None:
    from llama_index.core.base.response.schema import Response
    from llama_index.core.query_engine import CustomQueryEngine
    from llama_index.core.tools import QueryEngineTool

    calls = []

    class FlakyEngine(CustomQueryEngine):
        def custom_query(self, query_str: str) -> Response:
            calls.append(query_str)
            raise RuntimeError("upstream 429")

    tool = QueryEngineTool.from_defaults(FlakyEngine(), name="docs", description="docs")

    output = call_tool(tool, {"input": "refund policy"})

    assert output.is_error
    assert calls == ["refund policy"]


@pytest.mark.asyncio
async def test_acall_tool_does_not_retry_failed_query_engine_tool() -> None:
    from llama_index.core.base.response.schema import Response
    from llama_index.core.query_engine import CustomQueryEngine
    from llama_index.core.tools import QueryEngineTool

    calls = []

    class FlakyEngine(CustomQueryEngine):
        async def acustom_query(self, query_str: str) -> Response:
            calls.append(query_str)
            raise RuntimeError("upstream 429")

        def custom_query(self, query_str: str) -> Response:
            raise AssertionError("async path should not hit the sync query")

    tool = QueryEngineTool.from_defaults(FlakyEngine(), name="docs", description="docs")

    output = await acall_tool(tool, {"input": "refund policy"})

    assert output.is_error
    assert calls == ["refund policy"]


def test_call_tool_does_not_retry_failed_retriever_tool() -> None:
    from typing import List

    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from llama_index.core.tools import RetrieverTool

    calls = []

    class FlakyRetriever(BaseRetriever):
        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            calls.append(query_bundle.query_str)
            raise RuntimeError("upstream 429")

    tool = RetrieverTool.from_defaults(
        FlakyRetriever(), name="docs", description="docs"
    )

    output = call_tool(tool, {"input": "refund policy"})

    assert output.is_error
    # RetrieverTool formats the positional input itself; what matters is
    # the retriever ran exactly once
    assert len(calls) == 1
    assert calls[0].startswith("refund policy")


def test_call_tool_supports_keyword_only_tool() -> None:
    from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput

    calls = []

    class KeywordOnlyTool(AsyncBaseTool):
        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(name="kw", description="kw")

        def call(self, *, input: str) -> ToolOutput:
            calls.append(input)
            return ToolOutput(
                content=input,
                tool_name="kw",
                raw_input={"input": input},
                raw_output=input,
            )

        async def acall(self, *, input: str) -> ToolOutput:
            return self.call(input=input)

    output = call_tool(KeywordOnlyTool(), {"input": "charge"})

    assert not output.is_error
    assert calls == ["charge"]


@pytest.mark.asyncio
async def test_acall_tool_supports_keyword_only_tool() -> None:
    from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput

    calls = []

    class KeywordOnlyTool(AsyncBaseTool):
        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(name="kw", description="kw")

        def call(self, *, input: str) -> ToolOutput:
            raise AssertionError("async path should not hit the sync call")

        async def acall(self, *, input: str) -> ToolOutput:
            calls.append(input)
            return ToolOutput(
                content=input,
                tool_name="kw",
                raw_input={"input": input},
                raw_output=input,
            )

    output = await acall_tool(KeywordOnlyTool(), {"input": "charge"})

    assert not output.is_error
    assert calls == ["charge"]


@pytest.mark.asyncio
async def test_acall_tool_supports_keyword_only_plain_tool() -> None:
    from llama_index.core.tools.types import BaseTool, ToolMetadata, ToolOutput

    calls = []

    class PlainKeywordOnlyTool(BaseTool):
        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(name="kw", description="kw")

        def __call__(self, *, input: str) -> ToolOutput:
            calls.append(input)
            return ToolOutput(
                content=input,
                tool_name="kw",
                raw_input={"input": input},
                raw_output=input,
            )

    output = await acall_tool(PlainKeywordOnlyTool(), {"input": "charge"})

    assert not output.is_error
    assert calls == ["charge"]


def test_predict_and_call_does_not_double_run_failed_engine() -> None:
    from typing import Any, List

    from llama_index.core.base.response.schema import Response
    from llama_index.core.llms import ChatResponse
    from llama_index.core.llms.llm import ToolSelection
    from llama_index.core.llms.mock import MockFunctionCallingLLM
    from llama_index.core.query_engine import CustomQueryEngine
    from llama_index.core.tools import QueryEngineTool

    calls = []

    class FlakyEngine(CustomQueryEngine):
        def custom_query(self, query_str: str) -> Response:
            calls.append(query_str)
            raise RuntimeError("upstream 429")

    class OneToolCallLLM(MockFunctionCallingLLM):
        def get_tool_calls_from_response(
            self,
            response: ChatResponse,
            error_on_no_tool_call: bool = False,
            **kwargs: Any,
        ) -> List[ToolSelection]:
            return [
                ToolSelection(
                    tool_id="c1",
                    tool_name="docs",
                    tool_kwargs={"input": "refund policy"},
                )
            ]

    tool = QueryEngineTool.from_defaults(FlakyEngine(), name="docs", description="docs")

    OneToolCallLLM().predict_and_call([tool], user_msg="refund?")

    assert calls == ["refund policy"]


@pytest.mark.asyncio
async def test_apredict_and_call_does_not_double_run_failed_engine() -> None:
    from typing import Any, List

    from llama_index.core.base.response.schema import Response
    from llama_index.core.llms import ChatResponse
    from llama_index.core.llms.llm import ToolSelection
    from llama_index.core.llms.mock import MockFunctionCallingLLM
    from llama_index.core.query_engine import CustomQueryEngine
    from llama_index.core.tools import QueryEngineTool

    calls = []

    class FlakyEngine(CustomQueryEngine):
        async def acustom_query(self, query_str: str) -> Response:
            calls.append(query_str)
            raise RuntimeError("upstream 429")

        def custom_query(self, query_str: str) -> Response:
            raise AssertionError("async path should not hit the sync query")

    class OneToolCallLLM(MockFunctionCallingLLM):
        def get_tool_calls_from_response(
            self,
            response: ChatResponse,
            error_on_no_tool_call: bool = False,
            **kwargs: Any,
        ) -> List[ToolSelection]:
            return [
                ToolSelection(
                    tool_id="c1",
                    tool_name="docs",
                    tool_kwargs={"input": "refund policy"},
                )
            ]

    tool = QueryEngineTool.from_defaults(FlakyEngine(), name="docs", description="docs")

    await OneToolCallLLM().apredict_and_call([tool], user_msg="refund?")

    assert calls == ["refund policy"]
