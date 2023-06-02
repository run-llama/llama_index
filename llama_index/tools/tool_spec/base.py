"""Base tool spec class."""

from abc import ABC, abstractmethod
from typing import List, Optional
from llama_index.tools.types import ToolMetadata
from llama_index.tools.function_tool import FunctionTool
from inspect import signature


class BaseToolSpec:
    """Base tool spec class."""

    # list of functions that you'd want to convert to spec
    spec_functions: List[str]

    @classmethod
    def to_tool_list(
        cls,
        func_to_metadata_mapping: Optional[ToolMetadata] = None,
    ) -> List[FunctionTool]:
        """Convert tool spec to list of tools."""
        func_to_metadata_mapping = func_to_metadata_mapping or {}
        tool_list = []
        for func_name in cls.spec_functions:
            func = getattr(cls, func_name)
            metadata = func_to_metadata_mapping.get(func_name, None)
            if metadata is None:
                name = func_name
                docstring = func.__doc__
                description = f"{name}{signature(func)}\n{docstring}"
                metadata = ToolMetadata(name=name, description=description)
            tool = FunctionTool.from_defaults(fn=func, metadata=metadata)
            tool_list.append(tool)
        return tool_list
