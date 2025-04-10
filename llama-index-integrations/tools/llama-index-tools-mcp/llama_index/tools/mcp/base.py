from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field, create_model
from mcp.client.session import ClientSession

import asyncio

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import ToolMetadata

# Map JSON Schema types to Python types
json_type_mapping: Dict[str, Type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def create_model_from_json_schema(
    schema: Dict[str, Any], model_name: str = "DynamicModel"
) -> Type[BaseModel]:
    """
    To create a Pydantic model from the JSON Schema of MCP tools.

    Args:
        schema: A JSON Schema dictionary containing properties and required fields.
        model_name: The name of the model.

    Returns:
        A Pydantic model class.
    """
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    fields = {}

    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        json_type = json_type[0] if isinstance(json_type, list) else json_type

        field_type = json_type_mapping.get(json_type, str)
        if field_name in required_fields:
            default_value = ...
        else:
            default_value = None
            field_type = Optional[field_type]
        fields[field_name] = (
            field_type,
            Field(default_value, description=field_schema.get("description", "")),
        )
    return create_model(model_name, **fields)


class McpToolSpec(BaseToolSpec):
    """
    MCPToolSpec will get the tools from MCP Client (only need to implement ClientSession) and convert them to LlamaIndex's FunctionTool objects.

    Args:
        client: An MCP client instance implementing ClientSession, and it should support the following methods in ClientSession:
            - list_tools: List all tools.
            - call_tool: Call a tool.
        allowed_tools: If set, only return tools with the specified names.
    """

    def __init__(
        self,
        client: ClientSession,
        allowed_tools: Optional[List[str]] = None,
    ) -> None:
        self.client = client
        self.allowed_tools = allowed_tools if allowed_tools is not None else []

    async def fetch_tools(self) -> List[Any]:
        """
        An asynchronous method to get the tools list from MCP Client. If allowed_tools is set, it will filter the tools.

        Returns:
            A list of tools, each tool object needs to contain name, description, inputSchema properties.
        """
        response = await self.client.list_tools()
        tools = response.tools if hasattr(response, "tools") else []
        if self.allowed_tools:
            tools = [tool for tool in tools if tool.name in self.allowed_tools]
        return tools

    def _create_tool_fn(self, tool_name: str) -> Callable:
        """
        Create a tool call function for a specified MCP tool name. The function internally wraps the call_tool call to the MCP Client.
        """

        async def async_tool_fn(**kwargs):
            return await self.client.call_tool(tool_name, kwargs)

        return async_tool_fn

    async def to_tool_list_async(self) -> List[FunctionTool]:
        """
        Asynchronous method to convert MCP tools to FunctionTool objects.

        Returns:
            A list of FunctionTool objects.
        """
        tools_list = await self.fetch_tools()
        function_tool_list: List[FunctionTool] = []
        for tool in tools_list:
            fn = self._create_tool_fn(tool.name)
            # Create a Pydantic model based on the tool inputSchema
            model_schema = create_model_from_json_schema(
                tool.inputSchema, model_name=f"{tool.name}_Schema"
            )
            metadata = ToolMetadata(
                name=tool.name,
                description=tool.description,
                fn_schema=model_schema,
            )
            function_tool = FunctionTool.from_defaults(fn=fn, tool_metadata=metadata)
            function_tool_list.append(function_tool)
        return function_tool_list

    def to_tool_list(self) -> List[FunctionTool]:
        """
        Synchronous interface: Convert MCP Client tools to FunctionTool objects.
        Note: This method should not be called in an asynchronous environment, otherwise an exception will be thrown. Use to_tool_list_async instead.

        Returns:
            A list of FunctionTool objects.
        """
        return patch_sync(self.to_tool_list_async)()


def patch_sync(func_async: Callable) -> Callable:
    def patched_sync(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        # If the current environment is asynchronous, raise an exception to prompt the use of the asynchronous interface
        if loop and loop.is_running():
            raise RuntimeError(
                "In an asynchronous environment, synchronous calls are not supported. Please use the asynchronous interface (e.g., to_tool_list_async) instead."
            )
        return asyncio.run(func_async(*args, **kwargs))

    return patched_sync
