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
from llama_index.core.schema import Document
from llama_index.core.tools.function_tool import FunctionTool
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


def test_ondemand_loader_tool_default_name_and_from_tool(
    patch_llm_predictor,
) -> None:
    """Test default name, from_tool metadata inheritance, and required query_str."""
    reader = StringIterableReader()
    default_tool = OnDemandLoaderTool.from_defaults(
        reader=reader,
        index_cls=VectorStoreIndex,
    )
    assert default_tool.metadata.get_name() == "LoadData"
    schema = default_tool.metadata.get_parameters_dict()
    assert "query_str" in schema.get("required", [])
    out = default_tool(["Hello world."], query_str="What is?")
    assert out.tool_name == "LoadData"
    assert str(out) == "What is?:Hello world."

    def load_docs(texts: List[str]) -> List[Document]:
        """Load documents from texts."""
        return [Document(text=t) for t in texts]

    fn_tool = FunctionTool.from_defaults(
        fn=load_docs,
        name="custom_doc_loader",
        description="Custom loader description",
    )
    wrapped = OnDemandLoaderTool.from_tool(
        tool=fn_tool,
        index_cls=VectorStoreIndex,
    )
    assert wrapped.metadata.get_name() == "custom_doc_loader"
    assert wrapped.metadata.description == "Custom loader description"
    wrapped_schema = wrapped.metadata.get_parameters_dict()
    assert "query_str" in wrapped_schema.get("required", [])
    wrapped_out = wrapped(["Hello world."], query_str="What is?")
    assert wrapped_out.tool_name == "custom_doc_loader"
    assert str(wrapped_out) == "What is?:Hello world."
