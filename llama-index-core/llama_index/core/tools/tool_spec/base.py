"""Base tool spec class."""

import asyncio
from inspect import signature
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import ToolMetadata

AsyncCallable = Callable[..., Awaitable[Any]]


# TODO: deprecate the Tuple (there's no use for it)
SPEC_FUNCTION_TYPE = Union[str, Tuple[str, str]]


class BaseToolSpec:
    """Base tool spec class."""

    # list of functions that you'd want to convert to spec
    spec_functions: List[SPEC_FUNCTION_TYPE]

    def get_fn_schema_from_fn_name(
        self, fn_name: str, spec_functions: Optional[List[SPEC_FUNCTION_TYPE]] = None
    ) -> Optional[Type[BaseModel]]:
        """
        NOTE: This function is deprecated and kept only for backwards compatibility.

        Return map from function name.

        Return type is Optional, meaning that the schema can be None.
        In this case, it's up to the downstream tool implementation to infer the schema.

        """
        return None

    def get_metadata_from_fn_name(
        self, fn_name: str, spec_functions: Optional[List[SPEC_FUNCTION_TYPE]] = None
    ) -> Optional[ToolMetadata]:
        """
        NOTE: This function is deprecated and kept only for backwards compatibility.

        Return map from function name.

        Return type is Optional, meaning that the schema can be None.
        In this case, it's up to the downstream tool implementation to infer the schema.

        """
        schema = self.get_fn_schema_from_fn_name(fn_name, spec_functions=spec_functions)
        if schema is None:
            return None

        func = getattr(self, fn_name)
        name = fn_name
        docstring = func.__doc__ or ""
        description = f"{name}{signature(func)}\n{docstring}"
        fn_schema = self.get_fn_schema_from_fn_name(
            fn_name, spec_functions=spec_functions
        )
        return ToolMetadata(name=name, description=description, fn_schema=fn_schema)

    def to_tool_list(
        self,
        spec_functions: Optional[List[SPEC_FUNCTION_TYPE]] = None,
        func_to_metadata_mapping: Optional[Dict[str, ToolMetadata]] = None,
    ) -> List[FunctionTool]:
        """Convert tool spec to list of tools."""
        spec_functions = spec_functions or self.spec_functions
        func_to_metadata_mapping = func_to_metadata_mapping or {}
        tool_list = []
        for func_spec in spec_functions:
            func_sync = None
            func_async = None
            if isinstance(func_spec, str):
                func = getattr(self, func_spec)
                if asyncio.iscoroutinefunction(func):
                    func_async = func
                else:
                    func_sync = func
                metadata = func_to_metadata_mapping.get(func_spec, None)
                if metadata is None:
                    metadata = self.get_metadata_from_fn_name(func_spec)
            elif isinstance(func_spec, tuple) and len(func_spec) == 2:
                func_sync = getattr(self, func_spec[0])
                func_async = getattr(self, func_spec[1])
                metadata = func_to_metadata_mapping.get(func_spec[0], None)
                if metadata is None:
                    metadata = func_to_metadata_mapping.get(func_spec[1], None)
                    if metadata is None:
                        metadata = self.get_metadata_from_fn_name(func_spec[0])
            else:
                raise ValueError(
                    "spec_functions must be of type: List[Union[str, Tuple[str, str]]]"
                )

            tool = FunctionTool.from_defaults(
                fn=func_sync,
                async_fn=func_async,
                tool_metadata=metadata,
            )
            tool_list.append(tool)
        return tool_list

    async def to_tool_list_async(
        self,
        spec_functions: Optional[List[SPEC_FUNCTION_TYPE]] = None,
        func_to_metadata_mapping: Optional[Dict[str, ToolMetadata]] = None,
    ) -> List[FunctionTool]:
        """Asynchronously convert a tool spec to a list of tools."""
        return await asyncio.to_thread(
            self.to_tool_list, spec_functions, func_to_metadata_mapping
        )
