from typing import Any

import pytest

from llama_index.core.bridge.pydantic import Field
from llama_index.core.tools.function_tool import FunctionTool


@pytest.mark.parametrize("async_fn", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        Field(default=5, description="Result limit"),
        Field(description="Result limit"),
        Field(default_factory=lambda: 5, description="Result limit"),
    ],
    ids=["default", "required", "factory"],
)
@pytest.mark.asyncio
async def test_field_after_regular_default(field: Any, async_fn: bool) -> None:
    def search(query: str = "topic", limit: int = field) -> str:
        return f"{query}:{limit}"

    async def asearch(query: str = "topic", limit: int = field) -> str:
        return f"{query}:{limit}"

    tool = (
        FunctionTool.from_defaults(async_fn=asearch)
        if async_fn
        else FunctionTool.from_defaults(search)
    )
    schema = tool.metadata.get_parameters_dict()
    assert schema["properties"]["query"]["default"] == "topic"
    assert schema["properties"]["limit"]["description"] == "Result limit"
    assert ("limit" in schema.get("required", [])) == field.is_required()
    assert "FieldInfo" not in tool.metadata.description

    assert tool.call(limit=7).content == "topic:7"
    assert (await tool.acall(limit=7)).content == "topic:7"
    if not field.is_required():
        assert tool.call().content == "topic:5"
        assert (await tool.acall()).content == "topic:5"
