"""Test ad-hoc loader Tool."""

from typing import List
from pydantic import BaseModel
from llama_index.readers.schema.base import Document
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.tools.ondemand_loader_tool import (
    OnDemandLoaderTool,
    create_schema_from_function,
)

from llama_index.readers.string_iterable import StringIterableReader


def test_create_schema_from_function() -> None:
    """Test create schema from function."""

    def test_fn(x: int, y: int, z: List[str]) -> None:
        """Test function."""
        pass

    SchemaCls = create_schema_from_function("test_schema", test_fn)
    schema = SchemaCls.schema()
    assert schema["properties"]["x"]["type"] == "integer"
    assert schema["properties"]["y"]["type"] == "integer"
    assert schema["properties"]["z"]["type"] == "array"

    SchemaCls = create_schema_from_function("test_schema", test_fn, [("a", bool, 1)])
    schema = SchemaCls.schema()
    assert schema["properties"]["a"]["type"] == "boolean"


def test_ondemand_loader_tool(
    mock_service_context: ServiceContext,
    documents: List[Document],
) -> None:
    """Test ondemand loader."""

    class TestSchemaSpec(BaseModel):
        """Test schema spec."""

        texts: List[str]
        query_str: str

    # import most basic string reader
    reader = StringIterableReader()
    tool = OnDemandLoaderTool.from_defaults(
        reader=reader,
        index_cls=VectorStoreIndex,
        index_kwargs={"service_context": mock_service_context},
        name="ondemand_loader_tool",
        description="ondemand_loader_tool_desc",
        fn_schema=TestSchemaSpec,
    )
    response = tool(["Hello world."], query_str="What is?")
    assert response == "What is?:Hello world."

    # convert tool to structured langchain tool
    lc_tool = tool.to_langchain_structured_tool()
    assert lc_tool.args_schema == TestSchemaSpec
    response = lc_tool.run({"texts": ["Hello world."], "query_str": "What is?"})
    assert response == "What is?:Hello world."
