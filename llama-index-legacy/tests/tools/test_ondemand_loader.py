"""Test ad-hoc loader Tool."""

from typing import List

import pytest

try:
    import langchain
except ImportError:
    langchain = None  # type: ignore

from llama_index.legacy.bridge.pydantic import BaseModel
from llama_index.legacy.indices.vector_store.base import VectorStoreIndex
from llama_index.legacy.readers.string_iterable import StringIterableReader
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.tools.ondemand_loader_tool import OnDemandLoaderTool


class TestSchemaSpec(BaseModel):
    """Test schema spec."""

    texts: List[str]
    query_str: str


@pytest.fixture()
def tool(mock_service_context: ServiceContext) -> OnDemandLoaderTool:
    # import most basic string reader
    reader = StringIterableReader()
    return OnDemandLoaderTool.from_defaults(
        reader=reader,
        index_cls=VectorStoreIndex,
        index_kwargs={"service_context": mock_service_context},
        name="ondemand_loader_tool",
        description="ondemand_loader_tool_desc",
        fn_schema=TestSchemaSpec,
    )


def test_ondemand_loader_tool(
    tool: OnDemandLoaderTool,
) -> None:
    """Test ondemand loader."""
    response = tool(["Hello world."], query_str="What is?")
    assert str(response) == "What is?:Hello world."


@pytest.mark.skipif(langchain is None, reason="langchain not installed")
def test_ondemand_loader_tool_langchain(
    tool: OnDemandLoaderTool,
) -> None:
    # convert tool to structured langchain tool
    lc_tool = tool.to_langchain_structured_tool()
    assert lc_tool.args_schema == TestSchemaSpec
    response = lc_tool.run({"texts": ["Hello world."], "query_str": "What is?"})
    assert str(response) == "What is?:Hello world."
