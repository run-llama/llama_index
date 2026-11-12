"""
A2A Tool Spec - converts A2A agent skills to LlamaIndex FunctionTool objects.

Mirrors the McpToolSpec pattern from llama-index-tools-mcp.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, create_model

from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.tools.types import ToolMetadata
from llama_index.tools.a2a.tool_spec_mixins import (
    TypeResolutionMixin,
    TypeCreationMixin,
    FieldExtractionMixin,
)


class A2AToolSpec(
    BaseToolSpec, TypeResolutionMixin, TypeCreationMixin, FieldExtractionMixin
):
    """
    A2AToolSpec converts A2A agent skills to LlamaIndex's FunctionTool objects.

    This spec connects to an A2A agent via its AgentCard and exposes
    the agent's capabilities as callable tools within LlamaIndex workflows.

    Args:
        client: An A2A client instance implementing basic communication methods.
        allowed_tools: If set, only return tools with the specified names.
        global_partial_params: A dict of params to apply to all tools globally.
        partial_params_by_tool: A dict mapping tool names to param overrides.
        include_capabilities: Whether to include agent capabilities in the tool list.

    """

    def __init__(
        self,
        client: "BasicA2AClient",  # type: ignore[name-defined]
        allowed_tools: Optional[List[str]] = None,
        global_partial_params: Optional[Dict[str, Any]] = None,
        partial_params_by_tool: Optional[Dict[str, Dict[str, Any]]] = None,
        include_capabilities: bool = False,
    ) -> None:
        self.client = client
        self.allowed_tools = allowed_tools
        self.global_partial_params = global_partial_params
        self.partial_params_by_tool = partial_params_by_tool
        self.include_capabilities = include_capabilities
        self.properties_cache = {}

    async def fetch_skills(self) -> List[Any]:
        """
        Asynchronously get the list of skills from the A2A agent.

        Returns:
            A list of skill objects, each containing name, description, input_schema.

        """
        response = await self.client.list_skills()
        skills = response if isinstance(response, list) else getattr(response, "skills", [])

        if self.allowed_tools is None:
            return skills

        if any(self.allowed_tools):
            return [skill for skill in skills if skill.name in self.allowed_tools]

        logging.warning(
            "Returning an empty tool list due to the empty `allowed_tools` list. "
            "Please ensure `allowed_tools` is set appropriately."
        )
        return []

    def _create_tool_fn(self, skill_name: str) -> Callable:
        """
        Create a tool call function for a specified A2A skill name.
        The function internally wraps the call_skill call to the A2A client.
        """
        async def async_tool_fn(**kwargs):
            return await self.client.call_skill(skill_name, kwargs)

        return async_tool_fn

    async def to_tool_list_async(self) -> List[FunctionTool]:
        """
        Asynchronous method to convert A2A skills to FunctionTool objects.

        Returns:
            A list of FunctionTool objects.

        """
        skills_list = await self.fetch_skills()
        function_tool_list: List[FunctionTool] = []

        for skill in skills_list:
            fn = self._create_tool_fn(skill.name)

            # Create a Pydantic model based on the skill input schema
            model_schema = self.create_model_from_json_schema(
                skill.input_schema, model_name=f"{skill.name}_Schema"
            )

            # Set up global partial params as default
            tool_partial_params = dict(self.global_partial_params or {})

            # Override with tool-specific partial params
            if self.partial_params_by_tool and skill.name in self.partial_params_by_tool:
                tool_overrides = self.partial_params_by_tool[skill.name]
                for key, value in tool_overrides.items():
                    if value is None:
                        tool_partial_params.pop(key, None)
                    else:
                        tool_partial_params[key] = value

            # Remove fields that are set as partial params
            if tool_partial_params:
                model_schema = self.remove_model_fields(
                    model_schema,
                    fields_to_remove=set(tool_partial_params),
                    model_name=f"{skill.name}_Schema",
                )

            metadata = ToolMetadata(
                name=skill.name,
                description=skill.description,
                fn_schema=model_schema,
            )
            function_tool = FunctionTool.from_defaults(
                async_fn=fn,
                tool_metadata=metadata,
                partial_params=tool_partial_params,
            )
            function_tool_list.append(function_tool)

        return function_tool_list

    def to_tool_list(self) -> List[FunctionTool]:
        """
        Synchronous interface: Convert A2A agent skills to FunctionTool objects.
        Note: Use to_tool_list_async instead in asynchronous environments.

        Returns:
            A list of FunctionTool objects.

        """
        return patch_sync(self.to_tool_list_async)()

    def create_model_from_json_schema(
        self,
        schema: dict[str, Any],
        model_name: str = "DynamicModel",
    ) -> type[BaseModel]:
        """
        Create a Pydantic model from a JSON Schema dictionary.

        Args:
            schema: A JSON Schema dictionary containing properties and required fields.
            model_name: The name of the model.

        Returns:
            A Pydantic model class.

        """
        defs = schema.get("$defs", {})

        # Process all type definitions
        for cls_name, cls_schema in defs.items():
            self.properties_cache[cls_name] = self._create_model(
                cls_schema,
                cls_name,
                defs,
            )

        return self._create_model(schema, model_name)

    def _create_model(
        self,
        schema: dict,
        model_name: str,
        defs: dict = {},
    ) -> type[BaseModel]:
        """Create a Pydantic model from a schema."""
        if model_name in self.properties_cache:
            return self.properties_cache[model_name]

        fields = self._extract_fields(schema, defs)
        model = create_model(model_name, **fields)
        self.properties_cache[model_name] = model
        return model

    def remove_model_fields(
        self,
        model: type[BaseModel],
        fields_to_remove: Set[str],
        model_name: str,
    ) -> type[BaseModel]:
        """Remove specified fields from a Pydantic model."""
        fields = {
            name: (
                field.annotation,
                field.default if field.is_required() else field.default,
            )
            for name, field in model.model_fields.items()
            if name not in fields_to_remove
        }
        return create_model(model_name, **fields)


def patch_sync(func_async: Callable) -> Callable:
    """Patch an async function to provide a synchronous interface."""
    def patched_sync(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError(
                "In an asynchronous environment, synchronous calls are not supported. "
                "Please use the asynchronous interface (e.g., to_tool_list_async) instead."
            )
        return asyncio.run(func_async(*args, **kwargs))

    return patched_sync
