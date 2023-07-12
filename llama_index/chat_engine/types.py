import logging
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, List, Optional

from llama_index.tools import ToolOutput
from llama_index.llms.base import ChatMessage, ChatResponseGen, ChatResponseAsyncGen
from llama_index.memory import BaseMemory

logger = logging.getLogger(__name__)


@dataclass
class AgentChatResponse:
    """Agent chat response."""

    response: str = ""
    sources: List[ToolOutput] = field(default_factory=list)

    def __str__(self) -> str:
        return self.response


@dataclass
class StreamingAgentChatResponse:
    """Streaming chat response to user and writing to chat history."""

    response: str = ""
    sources: List[ToolOutput] = field(default_factory=list)
    chat_stream: Optional[ChatResponseGen] = None
    achat_stream: Optional[ChatResponseAsyncGen] = None
    _queue: queue.Queue = queue.Queue()
    _is_done = False
    _is_function: Optional[bool] = None

    def __str__(self) -> str:
        if self._is_done and not self._queue.empty() and not self._is_function:
            for delta in self._queue.queue:
                self.response += delta
        return self.response

    def write_response_to_history(self, memory: BaseMemory) -> None:
        if self.chat_stream is None:
            raise ValueError(
                "chat_stream is None. Cannot write to history without chat_stream."
            )

        final_message = None
        for chat in self.chat_stream:
            final_message = chat.message
            self._is_function = (
                final_message.additional_kwargs.get("function_call", None) is not None
            )
            self._queue.put_nowait(chat.delta)

        if final_message is not None:
            memory.put(final_message)

        self._is_done = True

    async def awrite_response_to_history(self, memory: BaseMemory) -> None:
        if self.achat_stream is None:
            raise ValueError(
                "achat_stream is None. Cannot asynchronously write to "
                "history without achat_stream."
            )

        final_message = None
        async for chat in self.achat_stream:
            final_message = chat.message
            self._is_function = (
                final_message.additional_kwargs.get("function_call", None) is not None
            )
            self._queue.put_nowait(chat.delta)

        if final_message is not None:
            memory.put(final_message)

        self._is_done = True

    @property
    def response_gen(self) -> Generator[str, None, None]:
        while not self._is_done or not self._queue.empty():
            try:
                delta = self._queue.get(block=False)
                self.response += delta
                yield delta
            except queue.Empty:
                # Queue is empty, but we're not done yet
                continue


class BaseChatEngine(ABC):
    """Base Chat Engine."""

    @abstractmethod
    def reset(self) -> None:
        """Reset conversation state."""
        pass

    @abstractmethod
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Main chat interface."""
        pass

    @abstractmethod
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """Stream chat interface."""
        pass

    @abstractmethod
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Async version of main chat interface."""
        pass

    @abstractmethod
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
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

    @property
    @abstractmethod
    def chat_history(self) -> List[ChatMessage]:
        pass


class ChatMode(str, Enum):
    """Chat Engine Modes."""

    SIMPLE = "simple"
    """Corresponds to `SimpleChatEngine`.
    
    Chat with LLM, without making use of a knowledge base.
    """
    CONDENSE_QUESTION = "condense_question"
    """Corresponds to `CondenseQuestionChatEngine`.
    
    First generate a standalone question from conversation context and last message,
    then query the query engine for a response.
    """
    REACT = "react"
    """Corresponds to `ReActAgent`.
    
    Use a ReAct agent loop with query engine tools. 
    """
