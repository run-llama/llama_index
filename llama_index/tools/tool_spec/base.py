"""Base tool spec class."""


from typing import List, Optional, Dict, Type, Union, Tuple, Callable, Awaitable, Any
from pydantic import BaseModel
from llama_index.tools.types import ToolMetadata
from llama_index.tools.function_tool import FunctionTool
from inspect import signature

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from llama_index.tools.utils import create_schema_from_function
import asyncio

AsyncCallable = Callable[..., Awaitable[Any]]


class BaseToolSpec:
    """Base tool spec class."""

    # list of functions that you'd want to convert to spec
    spec_functions: List[Union[str, Tuple[str, str]]]

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
        func_sync = None
        func_async = None
        for func_spec in self.spec_functions:
            if isinstance(func_spec, str):
                func = getattr(self, func_spec)
                if asyncio.iscoroutinefunction(func):
                    func_async = func
                else:
                    func_sync = func
                metadata = func_to_metadata_mapping.get(func_spec, None)
                if metadata is None:
                    name = func_spec
                    docstring = func.__doc__ or ""
                    description = f"{name}{signature(func)}\n{docstring}"
                    fn_schema = self.get_fn_schema_from_fn_name(func_spec)
                    metadata = ToolMetadata(
                        name=name, description=description, fn_schema=fn_schema
                    )
            elif isinstance(func_spec, tuple) and len(func_spec) == 2:
                func_sync = getattr(self, func_spec[0])
                func_async = getattr(self, func_spec[1])
                metadata = func_to_metadata_mapping.get(func_spec[0], None)
                if metadata is None:
                    metadata = func_to_metadata_mapping.get(func_spec[1], None)
                    if metadata is None:
                        name = func_spec[0]
                        docstring = func_sync.__doc__ or ""
                        description = f"{name}{signature(func_sync)}\n{docstring}"
                        fn_schema = self.get_fn_schema_from_fn_name(func_spec[0])
                        metadata = ToolMetadata(
                            name=name, description=description, fn_schema=fn_schema
                        )
            else:
                raise ValueError(
                    "spec_functions must be of type: List[Union[str, Tuple[str, str]]]"
                )

            if func_sync is None:
                if func_async is not None:
                    func_sync = patch_sync(func_async)
                else:
                    raise ValueError(
                        f"Could not retrieve a function for spec: {func_spec}"
                    )

            tool = FunctionTool.from_defaults(
                fn=func_sync,
                name=metadata.name,
                description=metadata.description,
                fn_schema=metadata.fn_schema,
                async_fn=func_async,
            )
            tool_list.append(tool)
        return tool_list


def patch_sync(func_async: AsyncCallable) -> Callable:
    def patched_sync(*args, **kwargs) -> Any:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func_async(*args, **kwargs))

    return patched_sync
