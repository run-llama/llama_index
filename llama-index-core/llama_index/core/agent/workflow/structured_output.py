"""Structured Output Tool for AgentWorkflow.

This module provides a tool-based approach for generating structured output
from agent workflows without requiring an extra LLM call.
"""

import json
from typing import Any, Dict, Type

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput


STRUCTURED_OUTPUT_TOOL_NAME = "submit_final_response"


class StructuredOutputTool(AsyncBaseTool):
    """
    A tool that captures structured output from the agent.

    This tool is automatically injected when `output_cls` is provided to an agent.
    When the agent calls this tool, the tool arguments are parsed as the structured
    response, eliminating the need for an extra LLM call.

    Example:
        ```python
        class MyOutput(BaseModel):
            answer: str
            confidence: float

        # Tool is automatically created and injected
        tool = StructuredOutputTool.from_output_cls(MyOutput)
        ```
    """

    def __init__(
        self,
        output_cls: Type[BaseModel],
        name: str = STRUCTURED_OUTPUT_TOOL_NAME,
        description: str | None = None,
    ) -> None:
        """
        Initialize the StructuredOutputTool.

        Args:
            output_cls: The Pydantic model class for structured output.
            name: The tool name. Defaults to "submit_final_response".
            description: Optional custom description. If not provided,
                a description is auto-generated from the output_cls schema.
        """
        self._output_cls = output_cls

        if description is None:
            schema = output_cls.model_json_schema()
            schema_str = json.dumps(schema, indent=2)
            description = (
                f"Submit your final response in a structured format. "
                f"Call this tool when you have completed the task and are ready "
                f"to provide the final answer. The response must conform to this schema:\n"
                f"{schema_str}"
            )

        self._metadata = ToolMetadata(
            name=name,
            description=description,
            fn_schema=output_cls,
            return_direct=True,
        )

    @classmethod
    def from_output_cls(
        cls,
        output_cls: Type[BaseModel],
        name: str = STRUCTURED_OUTPUT_TOOL_NAME,
        description: str | None = None,
    ) -> "StructuredOutputTool":
        """
        Create a StructuredOutputTool from a Pydantic model class.

        Args:
            output_cls: The Pydantic model class for structured output.
            name: The tool name. Defaults to "submit_final_response".
            description: Optional custom description.

        Returns:
            A StructuredOutputTool instance.
        """
        return cls(output_cls=output_cls, name=name, description=description)

    @property
    def metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        return self._metadata

    @property
    def output_cls(self) -> Type[BaseModel]:
        """Return the output class."""
        return self._output_cls

    def call(self, **kwargs: Any) -> ToolOutput:
        """
        Synchronously process the structured output.

        The tool arguments are validated against the output_cls schema
        and returned as a ToolOutput with the structured data.

        Args:
            **kwargs: The structured output fields.

        Returns:
            ToolOutput containing the validated structured response.
        """
        try:
            validated = self._output_cls.model_validate(kwargs)
            validated_dict = validated.model_dump()
            return ToolOutput(
                content=json.dumps(validated_dict),
                tool_name=self.metadata.name,
                raw_input=kwargs,
                raw_output=validated_dict,
                is_error=False,
            )
        except Exception as e:
            return ToolOutput(
                content=f"Error validating structured output: {e}",
                tool_name=self.metadata.name,
                raw_input=kwargs,
                raw_output=str(e),
                is_error=True,
            )

    async def acall(self, **kwargs: Any) -> ToolOutput:
        """
        Asynchronously process the structured output.

        The tool arguments are validated against the output_cls schema
        and returned as a ToolOutput with the structured data.

        Args:
            **kwargs: The structured output fields.

        Returns:
            ToolOutput containing the validated structured response.
        """
        return self.call(**kwargs)


def is_structured_output_tool(tool: AsyncBaseTool) -> bool:
    """Check if a tool is a StructuredOutputTool."""
    return isinstance(tool, StructuredOutputTool)


def extract_structured_output_from_tool_result(
    tool_output: ToolOutput,
    output_cls: Type[BaseModel],
) -> Dict[str, Any] | None:
    """
    Extract structured output from a tool result.

    Args:
        tool_output: The tool output to extract from.
        output_cls: The expected output class.

    Returns:
        The structured output as a dictionary, or None if extraction fails.
    """
    if tool_output.is_error:
        return None

    raw_output = tool_output.raw_output
    if isinstance(raw_output, dict):
        return raw_output

    if isinstance(raw_output, BaseModel):
        return raw_output.model_dump()

    if isinstance(raw_output, str):
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return None

    return None
