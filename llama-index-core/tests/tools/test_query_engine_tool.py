"""Test tools."""

from typing import Type, cast

import pytest
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.query_engine.custom import CustomQueryEngine
from llama_index.core.tools.query_engine import QueryEngineTool


class MockQueryEngine(CustomQueryEngine):
    """Custom query engine."""

    def custom_query(self, query_str: str) -> str:
        """Query."""
        return "custom_" + query_str


def test_query_engine_tool() -> None:
    """Test query engine tool."""
    query_engine = MockQueryEngine()  # type: ignore[call-arg]

    query_tool = QueryEngineTool.from_defaults(query_engine)

    # make sure both input formats work given function schema that assumes defaults
    response = query_tool("hello world")
    assert str(response) == "custom_hello world"
    response = query_tool(input="foo")
    assert str(response) == "custom_foo"

    fn_schema_cls = cast(Type[BaseModel], query_tool.metadata.fn_schema)
    fn_schema_obj = cast(BaseModel, fn_schema_cls(input="bar"))
    response = query_tool(**fn_schema_obj.model_dump())
    assert str(response) == "custom_bar"

    # test resolve input errors
    query_tool = QueryEngineTool.from_defaults(query_engine)
    response = query_tool(tmp="hello world")
    assert str(response) == "custom_{'tmp': 'hello world'}"

    with pytest.raises(ValueError):
        query_tool = QueryEngineTool.from_defaults(
            query_engine, resolve_input_errors=False
        )
        response = query_tool(tmp="hello world")
