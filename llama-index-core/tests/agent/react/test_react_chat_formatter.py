from typing import List, Optional, Sequence

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.types import (
    BaseReasoningStep,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import MessageRole
from llama_index.core.tools import BaseTool


class MockReActChatFormatter(ReActChatFormatter):
    def format(
        self,
        tools: Sequence[BaseTool],
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        return [ChatMessage(role=MessageRole.SYSTEM, content="mock data!")]


def test_inheritance_react_chat_formatter():
    formatter_from_defaults = MockReActChatFormatter.from_defaults()
    format_default_message = formatter_from_defaults.format([], [])
    assert format_default_message[0].content == "mock data!"

    formatter_from_context = MockReActChatFormatter.from_context("mock context!")
    format_context_message = formatter_from_context.format([], [])
    assert format_context_message[0].content == "mock data!"
