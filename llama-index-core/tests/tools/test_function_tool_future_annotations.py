"""FunctionTool context detection when annotations are strings (PEP 563)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.workflow import Context

if TYPE_CHECKING:
    from decimal import Decimal


class MyState(BaseModel):
    name: str = "Logan"


def test_ctx_param_detected_with_string_annotations() -> None:
    async def lookup(query: str, ctx: Context) -> str:
        """Look something up."""
        return query

    tool = FunctionTool.from_defaults(async_fn=lookup)
    assert tool.requires_context
    assert tool.ctx_param_name == "ctx"

    schema = tool.metadata.get_parameters_dict()
    assert list(schema["properties"]) == ["query"]


def test_ctx_param_excluded_from_description() -> None:
    def lookup(query: str, ctx: Context) -> str:
        """Look something up."""
        return query

    tool = FunctionTool.from_defaults(lookup)
    assert (
        tool.metadata.description == "lookup(query: 'str') -> 'str'\nLook something up."
    )


def test_generic_ctx_param_detected_with_string_annotations() -> None:
    async def lookup(query: str, ctx_arg: Context[MyState]) -> str:
        return query

    tool = FunctionTool.from_defaults(async_fn=lookup)
    assert tool.requires_context
    assert tool.ctx_param_name == "ctx_arg"
    assert list(tool.metadata.get_parameters_dict()["properties"]) == ["query"]


def test_unresolvable_annotation_falls_back() -> None:
    def total(amount: Decimal) -> str:
        return str(amount)

    tool = FunctionTool.from_defaults(total)
    assert not tool.requires_context
    assert tool.ctx_param_name is None
