from abc import abstractmethod
from typing import Sequence

from llama_index.chat_engine.types import BaseChatEngine
from llama_index.tools.types import BaseTool


class BaseChatAgent(BaseChatEngine):
    """A chat agent is a chat engine that makes use of tools."""

    @abstractmethod
    @classmethod
    def from_tools(
        cls,
        tools: Sequence[BaseTool],
        **kwargs,
    ) -> "BaseChatAgent":
        pass
