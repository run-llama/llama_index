import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from threading import Event
from typing import AsyncGenerator, Callable, Generator, List, Optional, Union, Dict, Any

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponseAsyncGen,
    ChatResponseGen,
)
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.core.memory import BaseMemory
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import ToolOutput
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatErrorEvent,
    StreamChatEndEvent,
    StreamChatStartEvent,
    StreamChatDeltaReceivedEvent,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def is_function(message: ChatMessage) -> bool:
    """Utility for ChatMessage responses from OpenAI models."""
    return "tool_calls" in message.additional_kwargs


class ChatResponseMode(str, Enum):
    """Flag toggling waiting/streaming in `Agent._chat`."""

    WAIT = "wait"
    STREAM = "stream"


@dataclass
class AgentChatResponse:
    """Agent chat response."""

    response: str = ""
    sources: List[ToolOutput] = field(default_factory=list)
    source_nodes: List[NodeWithScore] = field(default_factory=list)
    is_dummy_stream: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def set_source_nodes(self) -> None:
        if self.sources and not self.source_nodes:
            for tool_output in self.sources:
                if isinstance(tool_output.raw_output, (Response, StreamingResponse)):
                    self.source_nodes.extend(tool_output.raw_output.source_nodes)

    def __post_init__(self) -> None:
        self.set_source_nodes()

    def __str__(self) -> str:
        return self.response

    @property
    def response_gen(self) -> Generator[str, None, None]:
        """Used for fake streaming, i.e. with tool outputs."""
        if not self.is_dummy_stream:
            raise ValueError(
                "response_gen is only available for streaming responses. "
                "Set is_dummy_stream=True if you still want a generator."
            )

        for token in self.response.split(" "):
            yield token + " "
            time.sleep(0.1)

    async def async_response_gen(self) -> AsyncGenerator[str, None]:
        """Used for fake streaming, i.e. with tool outputs."""
        if not self.is_dummy_stream:
            raise ValueError(
                "response_gen is only available for streaming responses. "
                "Set is_dummy_stream=True if you still want a generator."
            )

        for token in self.response.split(" "):
            yield token + " "
            await asyncio.sleep(0.1)


@dataclass
class StreamingAgentChatResponse:
    """Streaming chat response to user and writing to chat history."""

    response: str = ""
    sources: List[ToolOutput] = field(default_factory=list)
    chat_stream: Optional[ChatResponseGen] = None
    achat_stream: Optional[ChatResponseAsyncGen] = None
    source_nodes: List[NodeWithScore] = field(default_factory=list)
    unformatted_response: str = ""
    queue: Queue = field(default_factory=Queue)
    aqueue: Optional[asyncio.Queue] = None
    # flag when chat message is a function call
    is_function: Optional[bool] = None
    # flag when processing done
    is_done = False
    # signal when a new item is added to the queue
    new_item_event: Optional[asyncio.Event] = None
    # NOTE: async code uses two events rather than one since it yields
    # control when waiting for queue item
    # signal when the OpenAI functions stop executing
    is_function_false_event: Optional[asyncio.Event] = None
    # signal when an OpenAI function is being executed
    is_function_not_none_thread_event: Event = field(default_factory=Event)
    # Track if an exception occurred
    exception: Optional[Exception] = None

    def set_source_nodes(self) -> None:
        if self.sources and not self.source_nodes:
            for tool_output in self.sources:
                if isinstance(tool_output.raw_output, (Response, StreamingResponse)):
                    self.source_nodes.extend(tool_output.raw_output.source_nodes)

    def __post_init__(self) -> None:
        self.set_source_nodes()

    def __str__(self) -> str:
        if self.is_done and not self.queue.empty() and not self.is_function:
            while self.queue.queue:
                delta = self.queue.queue.popleft()
                self.unformatted_response += delta
            self.response = self.unformatted_response.strip()
        return self.response

    def _ensure_async_setup(self) -> None:
        if self.aqueue is None:
            self.aqueue = asyncio.Queue()
        if self.new_item_event is None:
            self.new_item_event = asyncio.Event()
        if self.is_function_false_event is None:
            self.is_function_false_event = asyncio.Event()

    def put_in_queue(self, delta: Optional[str]) -> None:
        self.queue.put_nowait(delta)
        self.is_function_not_none_thread_event.set()

    def aput_in_queue(self, delta: Optional[str]) -> None:
        assert self.aqueue is not None
        assert self.new_item_event is not None

        self.aqueue.put_nowait(delta)
        self.new_item_event.set()

    @dispatcher.span
    def write_response_to_history(
        self,
        memory: BaseMemory,
        on_stream_end_fn: Optional[Callable] = None,
    ) -> None:
        if self.chat_stream is None:
            raise ValueError(
                "chat_stream is None. Cannot write to history without chat_stream."
            )

        # try/except to prevent hanging on error
        dispatcher.event(StreamChatStartEvent())
        try:
            final_text = ""
            for chat in self.chat_stream:
                self.is_function = is_function(chat.message)
                if chat.delta:
                    dispatcher.event(
                        StreamChatDeltaReceivedEvent(
                            delta=chat.delta,
                        )
                    )
                    self.put_in_queue(chat.delta)
                final_text += chat.delta or ""
            if self.is_function is not None:  # if loop has gone through iteration
                # NOTE: this is to handle the special case where we consume some of the
                # chat stream, but not all of it (e.g. in react agent)
                chat.message.content = final_text.strip()  # final message
                memory.put(chat.message)
        except Exception as e:
            dispatcher.event(StreamChatErrorEvent(exception=e))
            self.exception = e

            # This act as is_done events for any consumers waiting
            self.is_function_not_none_thread_event.set()

            # force the queue reader to see the exception
            self.put_in_queue("")
            raise
        dispatcher.event(StreamChatEndEvent())

        self.is_done = True

        # This act as is_done events for any consumers waiting
        self.is_function_not_none_thread_event.set()
        if on_stream_end_fn is not None and not self.is_function:
            on_stream_end_fn()

    @dispatcher.span
    async def awrite_response_to_history(
        self,
        memory: BaseMemory,
        on_stream_end_fn: Optional[Callable] = None,
    ) -> None:
        self._ensure_async_setup()
        assert self.aqueue is not None
        assert self.is_function_false_event is not None
        assert self.new_item_event is not None

        if self.achat_stream is None:
            raise ValueError(
                "achat_stream is None. Cannot asynchronously write to "
                "history without achat_stream."
            )

        # try/except to prevent hanging on error
        dispatcher.event(StreamChatStartEvent())
        try:
            final_text = ""
            async for chat in self.achat_stream:
                self.is_function = is_function(chat.message)
                if chat.delta:
                    dispatcher.event(
                        StreamChatDeltaReceivedEvent(
                            delta=chat.delta,
                        )
                    )
                    self.aput_in_queue(chat.delta)
                final_text += chat.delta or ""
                self.new_item_event.set()
                if self.is_function is False:
                    self.is_function_false_event.set()
            if self.is_function is not None:  # if loop has gone through iteration
                # NOTE: this is to handle the special case where we consume some of the
                # chat stream, but not all of it (e.g. in react agent)
                chat.message.content = final_text.strip()  # final message
                memory.put(chat.message)
        except Exception as e:
            dispatcher.event(StreamChatErrorEvent(exception=e))
            self.exception = e

            # These act as is_done events for any consumers waiting
            self.is_function_false_event.set()
            self.new_item_event.set()

            # force the queue reader to see the exception
            self.aput_in_queue("")
            raise
        dispatcher.event(StreamChatEndEvent())
        self.is_done = True

        # These act as is_done events for any consumers waiting
        self.is_function_false_event.set()
        self.new_item_event.set()
        if on_stream_end_fn is not None and not self.is_function:
            on_stream_end_fn()

    @property
    def response_gen(self) -> Generator[str, None, None]:
        while not self.is_done or not self.queue.empty():
            if self.exception is not None:
                raise self.exception

            try:
                delta = self.queue.get(block=False)
                self.unformatted_response += delta
                yield delta
            except Empty:
                # Queue is empty, but we're not done yet. Sleep for 0 secs to release the GIL and allow other threads to run.
                time.sleep(0)
        self.response = self.unformatted_response.strip()

    async def async_response_gen(self) -> AsyncGenerator[str, None]:
        self._ensure_async_setup()
        assert self.aqueue is not None

        while True:
            if not self.aqueue.empty() or not self.is_done:
                if self.exception is not None:
                    raise self.exception

                try:
                    delta = await asyncio.wait_for(self.aqueue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if self.is_done:
                        break
                    continue
                if delta is not None:
                    self.unformatted_response += delta
                    yield delta
            else:
                break
        self.response = self.unformatted_response.strip()

    def print_response_stream(self) -> None:
        for token in self.response_gen:
            print(token, end="", flush=True)

    async def aprint_response_stream(self) -> None:
        async for token in self.async_response_gen():
            print(token, end="", flush=True)


AGENT_CHAT_RESPONSE_TYPE = Union[AgentChatResponse, StreamingAgentChatResponse]


class BaseChatEngine(DispatcherSpanMixin, ABC):
    """Base Chat Engine."""

    @abstractmethod
    def reset(self) -> None:
        """Reset conversation state."""

    @abstractmethod
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Main chat interface."""

    @abstractmethod
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """Stream chat interface."""

    @abstractmethod
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Async version of main chat interface."""

    @abstractmethod
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """Async version of main chat interface."""

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

    def streaming_chat_repl(self) -> None:
        """Enter interactive chat REPL with streaming responses."""
        print("===== Entering Chat REPL =====")
        print('Type "exit" to exit.\n')
        self.reset()
        message = input("Human: ")
        while message != "exit":
            response = self.stream_chat(message)
            print("Assistant: ", end="", flush=True)
            response.print_response_stream()
            print("\n")
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

    CONTEXT = "context"
    """Corresponds to `ContextChatEngine`.

    First retrieve text from the index using the user's message, then use the context
    in the system prompt to generate a response.
    """

    CONDENSE_PLUS_CONTEXT = "condense_plus_context"
    """Corresponds to `CondensePlusContextChatEngine`.

    First condense a conversation and latest user message to a standalone question.
    Then build a context for the standalone question from a retriever,
    Then pass the context along with prompt and user message to LLM to generate a response.
    """

    REACT = "react"
    """Corresponds to `ReActAgent`.

    Use a ReAct agent loop with query engine tools.
    """

    OPENAI = "openai"
    """Corresponds to `OpenAIAgent`.

    Use an OpenAI function calling agent loop.

    NOTE: only works with OpenAI models that support function calling API.
    """

    BEST = "best"
    """Select the best chat engine based on the current LLM.

    Corresponds to `OpenAIAgent` if using an OpenAI model that supports
    function calling API, otherwise, corresponds to `ReActAgent`.
    """
