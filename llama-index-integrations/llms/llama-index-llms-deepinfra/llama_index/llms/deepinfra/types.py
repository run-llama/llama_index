from pydantic import BaseModel
from typing_extensions import Literal


class Function(BaseModel):
    name: str
    """The name of the function."""
    arguments: str
    """The arguments of the function."""


class ToolCallMessage(BaseModel):
    id: str
    """The ID of the tool call."""

    function: Function
    """The function that the model called."""

    type: Literal["function"]
    """The type of the tool. Currently, only `function` is supported."""
