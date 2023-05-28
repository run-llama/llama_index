import logging
from abc import ABC, abstractmethod
from enum import Enum

from llama_index.response.schema import RESPONSE_TYPE

logger = logging.getLogger(__name__)


class BaseChatEngine(ABC):
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
        print("===== Entering Chat REPL =====")
        print('Type "exit" to exit.\n')

        self.reset()
        message = input("Human: ")
        while message != "exit":
            response = self.chat(message)
            print(f"Assistant: {response}\n")
            message = input("Human: ")


class ChatMode(str, Enum):
    CONDENSE_QUESTION = "condense_question"
    REACT = "react"
    SIMPLE = "simple"
