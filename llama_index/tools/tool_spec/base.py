"""Base tool spec class."""

from typing import List, Optional, Dict
from llama_index.tools.types import ToolMetadata
from llama_index.tools.function_tool import FunctionTool
from inspect import signature


class BaseToolSpec:
    """Base tool spec class."""

    # list of functions that you'd want to convert to spec
    spec_functions: List[str]

    def to_tool_list(
        self,
        func_to_metadata_mapping: Optional[Dict[str, ToolMetadata]] = None,
    ) -> List[FunctionTool]:
        """Convert tool spec to list of tools."""
        func_to_metadata_mapping = func_to_metadata_mapping or {}
        tool_list = []
        for func_name in self.spec_functions:
            func = getattr(self, func_name)
            metadata = func_to_metadata_mapping.get(func_name, None)
            if metadata is None:
                name = func_name
                docstring = func.__doc__ or ""
                description = f"{name}{signature(func)}\n{docstring}"
                metadata = ToolMetadata(name=name, description=description)
            tool = FunctionTool(fn=func, metadata=metadata)
            tool_list.append(tool)
        return tool_list
