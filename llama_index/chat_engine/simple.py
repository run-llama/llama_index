import asyncio
from threading import Thread
from typing import Any, List, Type, Optional

from llama_index.chat_engine.types import (
    BaseChatEngine,
    StreamingChatResponse,
    STREAMING_CHAT_RESPONSE_TYPE,
)
from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.llms.base import LLM, ChatMessage
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.response.schema import RESPONSE_TYPE, Response, StreamingResponse


class SimpleChatEngine(BaseChatEngine):
    """Simple Chat Engine.

    Have a conversation with the LLM.
    This does not make use of a knowledge base.
    """

    def __init__(
        self,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages

    @classmethod
    def from_defaults(
        cls,
        service_context: Optional[ServiceContext] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> "SimpleChatEngine":
        """Initialize a SimpleChatEngine from default parameters."""
        service_context = service_context or ServiceContext.from_defaults()
        if not isinstance(service_context.llm_predictor, LLMPredictor):
            raise ValueError("llm_predictor must be a LLMPredictor instance")
        llm = service_context.llm_predictor.llm

        chat_history = chat_history or []
        memory = memory or memory_cls.from_defaults(chat_history=chat_history)

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        return cls(llm=llm, memory=memory, prefix_messages=prefix_messages)

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))
        all_messages = self._prefix_messages + self._memory.get()

        chat_response = self._llm.chat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return Response(response=chat_response.message.content)

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> STREAMING_CHAT_RESPONSE_TYPE:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))
        all_messages = self._prefix_messages + self._memory.get()

        chat_response = StreamingChatResponse(self._llm.stream_chat(all_messages))
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return StreamingResponse(response_gen=chat_response.response_gen)

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))
        all_messages = self._prefix_messages + self._memory.get()

        chat_response = await self._llm.achat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return Response(response=chat_response.message.content)

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> STREAMING_CHAT_RESPONSE_TYPE:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))
        all_messages = self._prefix_messages + self._memory.get()

        chat_response = StreamingChatResponse(self._llm.stream_chat(all_messages))
        thread = Thread(
            target=lambda x: asyncio.run(chat_response.awrite_response_to_history(x)),
            args=(self._memory,),
        )
        thread.start()

        return StreamingResponse(response_gen=chat_response.response_gen)

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
