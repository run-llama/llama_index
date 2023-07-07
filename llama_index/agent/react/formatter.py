# ReAct agent formatter

from abc import abstractmethod
from typing import List, Optional, Sequence

from pydantic import BaseModel

from llama_index.agent.react.prompts import (
    REACT_CHAT_LAST_USER_MESSAGE,
    REACT_CHAT_SYSTEM_HEADER,
)
from llama_index.agent.react.types import BaseReasoningStep
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.tools import BaseTool


def get_react_tool_descriptions(tools: Sequence[BaseTool]) -> List[str]:
    """Tool"""
    tool_descs = []
    for tool in tools:
        tool_desc = (
            f"> Tool Name: {tool.metadata.name}\n"
            f"Tool Description: {tool.metadata.description}\n"
            f"Tool Args: {tool.metadata.fn_schema_str}\n"
        )
        tool_descs.append(tool_desc)
    return tool_descs


# TODO: come up with better name
class BaseAgentChatFormatter(BaseModel):
    """Base chat formatter."""

    tools: Sequence[BaseTool]

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def format(
        self,
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""


class ReActChatFormatter(BaseAgentChatFormatter):
    """ReAct chat formatter."""

    system_header: str = REACT_CHAT_SYSTEM_HEADER
    last_user_message: str = REACT_CHAT_LAST_USER_MESSAGE

    def format(
        self,
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""
        current_reasoning = current_reasoning or []
        current_reasoning_str = (
            "\n".join(r.get_content() for r in current_reasoning)
            if current_reasoning
            else "None"
        )

        tool_descs_str = "\n".join(get_react_tool_descriptions(self.tools))

        fmt_sys_header = self.system_header.format(
            tool_desc=tool_descs_str,
            tool_names=", ".join([tool.metadata.get_name() for tool in self.tools]),
        )
        prev_chat_history = chat_history[:-1]
        last_chat = chat_history[-1]

        fmt_last_user_message = self.last_user_message.format(
            new_message=last_chat.content,
            current_reasoning=current_reasoning_str,
        )

        formatted_chat = [
            ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
            *prev_chat_history,
            ChatMessage(role=MessageRole.USER, content=fmt_last_user_message),
        ]
        return formatted_chat
