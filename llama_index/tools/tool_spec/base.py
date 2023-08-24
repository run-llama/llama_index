"""Base tool spec class."""

from inspect import signature
from typing import Dict, List, Optional, Type

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from llama_index.tools.function_tool import FunctionTool
from llama_index.tools.types import ToolMetadata
from llama_index.tools.utils import create_schema_from_function


class BaseToolSpec:
    """Base tool spec class."""

    # list of functions that you'd want to convert to spec
    spec_functions: List[str]

    def get_fn_schema_from_fn_name(self, fn_name: str) -> Optional[Type[BaseModel]]:
        """Return map from function name.

        Return type is Optional, meaning that the schema can be None.
        In this case, it's up to the downstream tool implementation to infer the schema.

        """
        for fn in self.spec_functions:
            if fn == fn_name:
                return create_schema_from_function(fn_name, getattr(self, fn_name))

        raise ValueError(f"Invalid function name: {fn_name}")

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
                fn_schema = self.get_fn_schema_from_fn_name(func_name)
                metadata = ToolMetadata(name=name, description=description, fn_schema=fn_schema)
            tool = FunctionTool.from_defaults(
                fn=func,
                name=metadata.name,
                description=metadata.description,
                fn_schema=metadata.fn_schema,
            )
            tool_list.append(tool)
        return tool_list
