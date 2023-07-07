import logging
import queue
from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncGenerator, Generator, List, Optional, Union

from llama_index.llms.base import ChatMessage, ChatResponseGen, ChatResponseAsyncGen
from llama_index.response.schema import RESPONSE_TYPE, StreamingResponse

logger = logging.getLogger(__name__)


class StreamingChatResponse:
    """Streaming chat response to user and writing to chat history."""

    def __init__(
        self, chat_stream: Union[ChatResponseGen, ChatResponseAsyncGen]
    ) -> None:
        self._chat_stream = chat_stream
        self._queue: queue.Queue = queue.Queue()
        self._is_done = False
        self._is_function: Optional[bool] = None
        self.response_str = ""

    def __str__(self) -> str:
        if self._is_done and not self._queue.empty() and not self._is_function:
            for delta in self._queue.queue:
                self.response_str += delta
        return self.response_str

    def write_response_to_history(self, chat_history: List[ChatMessage]) -> None:
        if isinstance(self._chat_stream, AsyncGenerator):
            raise ValueError(
                "Cannot write to history with async generator in sync function."
            )

        final_message = None
        for chat in self._chat_stream:
            final_message = chat.message
            self._is_function = (
                final_message.additional_kwargs.get("function_call", None) is not None
            )
            self._queue.put_nowait(chat.delta)

        if final_message is not None:
            chat_history.append(final_message)

        self._is_done = True

    async def awrite_response_to_history(self, chat_history: List[ChatMessage]) -> None:
        if isinstance(self._chat_stream, Generator):
            raise ValueError(
                "Cannot write to history with sync generator in async function."
            )

        final_message = None
        async for chat in self._chat_stream:
            final_message = chat.message
            self._is_function = (
                final_message.additional_kwargs.get("function_call", None) is not None
            )
            self._queue.put_nowait(chat.delta)

        if final_message is not None:
            chat_history.append(final_message)

        self._is_done = True

    @property
    def response_gen(self) -> Generator[str, None, None]:
        while not self._is_done or not self._queue.empty():
            try:
                delta = self._queue.get(block=False)
                self.response_str += delta
                yield delta
            except queue.Empty:
                # Queue is empty, but we're not done yet
                continue


STREAMING_CHAT_RESPONSE_TYPE = Union[
    StreamingResponse,
    StreamingChatResponse,
    Generator[StreamingChatResponse, None, None],
    AsyncGenerator[StreamingChatResponse, None],
]


class BaseChatEngine(ABC):
    """Base Chat Engine."""

    @abstractmethod
    def reset(self) -> None:
        """Reset conversation state."""
        pass

    @abstractmethod
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        """Main chat interface."""
        pass

    @abstractmethod
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> STREAMING_CHAT_RESPONSE_TYPE:
        """Stream chat interface."""
        pass

    @abstractmethod
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        """Async version of main chat interface."""
        pass

    @abstractmethod
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> STREAMING_CHAT_RESPONSE_TYPE:
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
    """Corresponds to `ReActChatEngine`.
    
    Use a ReAct agent loop with query engine tools. 
    Implemented via LangChain agent.
    """
