import asyncio
from typing import Any, List, Optional, Type

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.settings import Settings
from llama_index.core.types import Thread


class SimpleChatEngine(BaseChatEngine):
    """
    Simple Chat Engine.

    Have a conversation with the LLM.
    This does not make use of a knowledge base.
    """

    def __init__(
        self,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self.callback_manager = callback_manager or CallbackManager([])

    @classmethod
    def from_defaults(
        cls,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> "SimpleChatEngine":
        """Initialize a SimpleChatEngine from default parameters."""
        llm = llm or Settings.llm

        chat_history = chat_history or []
        memory = memory or memory_cls.from_defaults(chat_history=chat_history, llm=llm)

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [
                ChatMessage(content=system_prompt, role=llm.metadata.system_role)
            ]

        prefix_messages = prefix_messages or []

        return cls(
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            callback_manager=Settings.callback_manager,
        )

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        if hasattr(self._memory, "tokenizer_fn"):
            initial_token_count = len(
                self._memory.tokenizer_fn(
                    " ".join(
                        [
                            (m.content or "")
                            for m in self._prefix_messages
                            if isinstance(m.content, str)
                        ]
                    )
                )
            )
        else:
            initial_token_count = 0

        all_messages = self._prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = self._llm.chat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return AgentChatResponse(response=str(chat_response.message.content))

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        if hasattr(self._memory, "tokenizer_fn"):
            initial_token_count = len(
                self._memory.tokenizer_fn(
                    " ".join(
                        [
                            (m.content or "")
                            for m in self._prefix_messages
                            if isinstance(m.content, str)
                        ]
                    )
                )
            )
        else:
            initial_token_count = 0

        all_messages = self._prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(all_messages)
        )
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return chat_response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        if chat_history is not None:
            await self._memory.aset(chat_history)
        await self._memory.aput(ChatMessage(content=message, role="user"))

        if hasattr(self._memory, "tokenizer_fn"):
            initial_token_count = len(
                self._memory.tokenizer_fn(
                    " ".join(
                        [
                            (m.content or "")
                            for m in self._prefix_messages
                            if isinstance(m.content, str)
                        ]
                    )
                )
            )
        else:
            initial_token_count = 0

        all_messages = self._prefix_messages + (
            await self._memory.aget(initial_token_count=initial_token_count)
        )

        chat_response = await self._llm.achat(all_messages)
        ai_message = chat_response.message
        await self._memory.aput(ai_message)

        return AgentChatResponse(response=str(chat_response.message.content))

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            await self._memory.aset(chat_history)
        await self._memory.aput(ChatMessage(content=message, role="user"))

        if hasattr(self._memory, "tokenizer_fn"):
            initial_token_count = len(
                self._memory.tokenizer_fn(
                    " ".join(
                        [
                            (m.content or "")
                            for m in self._prefix_messages
                            if isinstance(m.content, str)
                        ]
                    )
                )
            )
        else:
            initial_token_count = 0

        all_messages = self._prefix_messages + (
            await self._memory.aget(initial_token_count=initial_token_count)
        )

        chat_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(all_messages)
        )
        chat_response.awrite_response_to_history_task = asyncio.create_task(
            chat_response.awrite_response_to_history(self._memory)
        )

        return chat_response

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
