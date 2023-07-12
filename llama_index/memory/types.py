from abc import abstractmethod
from pydantic import BaseModel
from typing import List

from llama_index.llms.base import ChatMessage


class BaseMemory(BaseModel):
    """Base class for all memory types."""

    @abstractmethod
    def get(self) -> List[ChatMessage]:
        """Get chat history."""
        pass

    @abstractmethod
    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        pass
