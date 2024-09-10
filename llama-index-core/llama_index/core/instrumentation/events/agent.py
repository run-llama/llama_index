from typing import Any, Optional

from llama_index.core.base.agent.types import TaskStepOutput, TaskStep
from llama_index.core.bridge.pydantic import model_validator, field_validator
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.tools.types import ToolMetadata


class AgentRunStepStartEvent(BaseEvent):
    """AgentRunStepStartEvent.

    Args:
        task_id (str): Task ID.
        step (Optional[TaskStep]): Task step.
        input (Optional[str]): Optional input.
    """

    task_id: str
    step: Optional[TaskStep]
    input: Optional[str]

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "AgentRunStepStartEvent"


class AgentRunStepEndEvent(BaseEvent):
    """AgentRunStepEndEvent.

    Args:
        step_output (TaskStepOutput): Task step output.
    """

    step_output: TaskStepOutput

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "AgentRunStepEndEvent"


class AgentChatWithStepStartEvent(BaseEvent):
    """AgentChatWithStepStartEvent.

    Args:
        user_msg (str): User input message.
    """

    user_msg: str

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "AgentChatWithStepStartEvent"


class AgentChatWithStepEndEvent(BaseEvent):
    """AgentChatWithStepEndEvent.

    Args:
        response (Optional[AGENT_CHAT_RESPONSE_TYPE]): Agent chat response.
    """

    response: Optional[AGENT_CHAT_RESPONSE_TYPE]

    @model_validator(mode="before")
    @classmethod
    def validate_response(cls: Any, values: Any) -> Any:
        """Validate response."""
        response = values.get("response")
        if response is None:
            pass
        elif not isinstance(response, AgentChatResponse) and not isinstance(
            response, StreamingAgentChatResponse
        ):
            raise ValueError(
                "response must be of type AgentChatResponse or StreamingAgentChatResponse"
            )

        return values

    @field_validator("response", mode="before")
    @classmethod
    def validate_response_type(cls: Any, response: Any) -> Any:
        """Validate response type."""
        if response is None:
            return response
        if not isinstance(response, AgentChatResponse) and not isinstance(
            response, StreamingAgentChatResponse
        ):
            raise ValueError(
                "response must be of type AgentChatResponse or StreamingAgentChatResponse"
            )
        return response

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "AgentChatWithStepEndEvent"


class AgentToolCallEvent(BaseEvent):
    """AgentToolCallEvent.

    Args:
        arguments (str): Arguments.
        tool (ToolMetadata): Tool metadata.
    """

    arguments: str
    tool: ToolMetadata

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "AgentToolCallEvent"
