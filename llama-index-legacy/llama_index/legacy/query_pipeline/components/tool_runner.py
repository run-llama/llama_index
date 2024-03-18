"""Tool runner component."""

from typing import Any, Dict, Optional, Sequence, cast

from llama_index.legacy.bridge.pydantic import Field
from llama_index.legacy.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
)
from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.core.query_pipeline.query_component import (
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.legacy.tools import AsyncBaseTool, adapt_to_async_tool


class ToolRunnerComponent(QueryComponent):
    """Tool runner component that takes in a set of tools."""

    tool_dict: Dict[str, AsyncBaseTool] = Field(
        ..., description="Dictionary of tool names to tools."
    )
    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )

    def __init__(
        self,
        tools: Sequence[AsyncBaseTool],
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        # determine parameters
        tool_dict = {tool.metadata.name: adapt_to_async_tool(tool) for tool in tools}
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            tool_dict=tool_dict, callback_manager=callback_manager, **kwargs
        )

    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self.callback_manager = callback_manager

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "tool_name" not in input:
            raise ValueError("tool_name must be provided in input")

        input["tool_name"] = validate_and_convert_stringable(input["tool_name"])

        if "tool_input" not in input:
            raise ValueError("tool_input must be provided in input")
        # make sure tool_input is a dictionary
        if not isinstance(input["tool_input"], dict):
            raise ValueError("tool_input must be a dictionary")

        return input

    def _run_component(self, **kwargs: Any) -> Dict:
        """Run component."""
        tool_name = kwargs["tool_name"]
        tool_input = kwargs["tool_input"]
        tool = cast(AsyncBaseTool, self.tool_dict[tool_name])
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: tool_input,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            tool_output = tool(**tool_input)
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        return {"output": tool_output}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        tool_name = kwargs["tool_name"]
        tool_input = kwargs["tool_input"]
        tool = cast(AsyncBaseTool, self.tool_dict[tool_name])
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: tool_input,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            tool_output = await tool.acall(**tool_input)
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        return {"output": tool_output}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"tool_name", "tool_input"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})
