"""Test tools built from functions with postponed (string) annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, List, Optional

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.workflow import Context

if TYPE_CHECKING:
    from decimal import Decimal


class Item(BaseModel):
    name: str


def add_item(item: Item, count: Annotated[int, "How many to add"] = 1) -> str:
    return f"{count} x {item.name}"


async def add_item_with_ctx(ctx: Context, item: Item) -> str:
    return item.name


def type_checking_only(amount: Decimal) -> str:
    return str(amount)


def list_tags(tags: Optional[List[str]] = None) -> str:
    return ",".join(tags or [])


def test_create_schema_from_function_postponed_annotations() -> None:
    schema = create_schema_from_function("AddItem", add_item).model_json_schema()

    assert schema["required"] == ["item"]
    assert schema["properties"]["item"]["$ref"] == "#/$defs/Item"
    assert schema["$defs"]["Item"]["properties"]["name"]["type"] == "string"
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["properties"]["count"]["description"] == "How many to add"
    assert schema["properties"]["count"]["default"] == 1


def test_create_schema_from_function_postponed_optional_list() -> None:
    schema = create_schema_from_function("ListTags", list_tags).model_json_schema()

    tags = schema["properties"]["tags"]
    assert tags["anyOf"] == [
        {"items": {"type": "string"}, "type": "array"},
        {"type": "null"},
    ]


def test_function_tool_ctx_param_postponed_annotations() -> None:
    tool = FunctionTool.from_defaults(async_fn=add_item_with_ctx)

    assert tool.requires_context
    assert tool.ctx_param_name == "ctx"
    schema = tool.metadata.get_parameters_dict()
    assert "ctx" not in schema["properties"]
    assert schema["required"] == ["item"]
    assert schema["properties"]["item"]["$ref"] == "#/$defs/Item"


def test_function_tool_unresolvable_postponed_annotations() -> None:
    tool = FunctionTool.from_defaults(type_checking_only)

    assert not tool.requires_context
    assert tool.metadata.name == "type_checking_only"
