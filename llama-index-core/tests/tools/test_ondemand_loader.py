"""Test ad-hoc loader Tool."""

import threading
from typing import List
from unittest.mock import MagicMock

import pytest

try:
    import langchain  # pants: no-infer-dep
except ImportError:
    langchain = None  # type: ignore

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.readers.string_iterable import StringIterableReader
from llama_index.core.schema import Document
from llama_index.core.tools import ToolMetadata
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


@pytest.mark.asyncio
async def test_async_load_and_index_build_run_off_the_event_loop() -> None:
    loop_thread = threading.get_ident()
    threads = []

    def load() -> List[Document]:
        threads.append(threading.get_ident())
        return [Document(text="loaded data")]

    def build(documents: List[Document]):
        threads.append(threading.get_ident())
        assert documents[0].text == "loaded data"
        return index

    async def answer(query: str) -> str:
        assert threading.get_ident() == loop_thread
        return query

    def as_query_engine():
        threads.append(threading.get_ident())
        return query_engine

    query_engine = MagicMock()
    query_engine.aquery = answer
    index = MagicMock()
    index.as_query_engine.side_effect = as_query_engine
    builder = MagicMock()
    builder.from_documents.side_effect = build
    tool = OnDemandLoaderTool(
        loader=load,
        index_cls=builder,
        index_kwargs={},
        metadata=ToolMetadata(name="load", description="Load test data"),
    )

    response = await tool.acall(query_str="query")
    assert str(response) == "query"
    assert len(threads) == 3
    assert loop_thread not in threads
