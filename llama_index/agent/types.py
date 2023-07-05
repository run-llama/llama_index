"""Base agent type."""
from typing import List
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms.base import ChatMessage
from abc import abstractmethod


# TODO: improve
CHAT_HISTORY_TYPE = List[ChatMessage]


class BaseAgent(BaseChatEngine, BaseQueryEngine):
    """Base Agent."""

    @abstractmethod
    def chat_history(self) -> CHAT_HISTORY_TYPE:
        """Chat history."""
