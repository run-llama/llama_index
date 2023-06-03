"""Test ad-hoc loader Tool."""

from typing import List
from pydantic import BaseModel
from llama_index.readers.schema.base import Document
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.tools.adhoc_loader_tool import AdhocLoaderTool

from llama_index.readers.string_iterable import StringIterableReader


def test_adhoc_loader_tool(
    mock_service_context: ServiceContext,
    documents: List[Document],
) -> None:
    """Test adhoc loader."""

    class TestSchemaSpec(BaseModel):
        """Test schema spec."""

        texts: List[str]
        query_str: str

    # import most basic string reader
    reader = StringIterableReader()
    tool = AdhocLoaderTool.from_defaults(
        reader=reader,
        index_cls=GPTVectorStoreIndex,
        index_kwargs={"service_context": mock_service_context},
        name="adhoc_loader_tool",
        description="adhoc_loader_tool_desc",
        fn_schema=TestSchemaSpec,
    )
    response = tool(["Hello world."], query_str="What is?")
    assert response == "What is?:Hello world."

    # convert tool to structured langchain tool
    lc_tool = tool.to_langchain_structured_tool()
    assert lc_tool.args_schema == TestSchemaSpec
    response = lc_tool.run({"texts": ["Hello world."], "query_str": "What is?"})
    assert response == "What is?:Hello world."
