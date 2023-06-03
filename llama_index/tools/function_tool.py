from typing import Any, Optional, Callable, Type

from pydantic import BaseModel
from llama_index.tools.types import BaseTool, ToolMetadata
from inspect import signature


class FunctionTool(BaseTool):
    """Function Tool.

    A tool that takes in a function.

    """

    def __init__(
        self,
        fn: Callable[..., Any],
        metadata: ToolMetadata,
    ) -> None:
        self._fn = fn
        self._metadata = metadata

    @classmethod
    def from_defaults(
        cls,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        fn_schema: Optional[Type[BaseModel]] = None,
    ) -> "FunctionTool":
        name = name or fn.__name__
        docstring = fn.__doc__
        description = description or f"{name}{signature(fn)}\n{docstring}"
        metadata = ToolMetadata(name=name, description=description, fn_schema=fn_schema)
        return cls(fn=fn, metadata=metadata)

    @property
    def metadata(self) -> ToolMetadata:
        """Metadata."""
        return self._metadata

    @property
    def fn(self) -> Callable[..., Any]:
        """Function."""
        return self._fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call."""
        return self._fn(*args, **kwargs)
