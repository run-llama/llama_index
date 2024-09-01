"""Test ad-hoc loader Tool."""

from typing import List

import pytest

try:
    import langchain  # pants: no-infer-dep
except ImportError:
    langchain = None  # type: ignore

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.readers.string_iterable import StringIterableReader
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool


class _TestSchemaSpec(BaseModel):
    """Test schema spec."""

    texts: List[str]
    query_str: str


@pytest.fixture()
def tool(patch_llm_predictor) -> OnDemandLoaderTool:
    # import most basic string reader
    reader = StringIterableReader()
    return OnDemandLoaderTool.from_defaults(
        reader=reader,
        index_cls=VectorStoreIndex,
        name="ondemand_loader_tool",
        description="ondemand_loader_tool_desc",
        fn_schema=_TestSchemaSpec,
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
    assert lc_tool.args_schema == _TestSchemaSpec
    response = lc_tool.run({"texts": ["Hello world."], "query_str": "What is?"})
    assert str(response) == "What is?:Hello world."
