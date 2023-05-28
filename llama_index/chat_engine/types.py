import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple

from llama_index.response.schema import RESPONSE_TYPE

logger = logging.getLogger(__name__)


class BaseChatEngine(ABC):
    """Base Chat Engine."""
    @abstractmethod
    def reset(self) -> None:
        """Reset conversation state."""
        pass

    @abstractmethod
    def chat(self, message: str) -> RESPONSE_TYPE:
        """Main chat interface."""
        pass

    @abstractmethod
    async def achat(self, message: str) -> RESPONSE_TYPE:
        """Async version of main chat interface."""
        pass

    def chat_repl(self) -> None:
        """Enter interactive chat REPL."""
        print("===== Entering Chat REPL =====")
        print('Type "exit" to exit.\n')

        self.reset()
        message = input("Human: ")
        while message != "exit":
            response = self.chat(message)
            print(f"Assistant: {response}\n")
            message = input("Human: ")


class ChatMode(str, Enum):
    """Chat Engine Models."""
    CONDENSE_QUESTION = "condense_question"
    REACT = "react"
    SIMPLE = "simple"


"""
Chat history is a list of (human_message, assistant_message) tuples.
"""
ChatHistoryType = List[Tuple[str, str]]
