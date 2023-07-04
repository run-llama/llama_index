from threading import Thread
from typing import Any, List, Optional

from llama_index.chat_engine.types import (
    BaseChatEngine,
    StreamingChatResponse,
)
from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.llms.base import LLM, ChatMessage
from llama_index.response.schema import RESPONSE_TYPE, Response, StreamingResponse


class SimpleChatEngine(BaseChatEngine):
    """Simple Chat Engine.

    Have a conversation with the LLM.
    This does not make use of a knowledge base.

    # TODO: add back ability to configure prompt/system message
    """

    def __init__(
        self,
        llm: LLM,
        chat_history: List[ChatMessage],
    ) -> None:
        self._llm = llm
        self._chat_history = chat_history

    @classmethod
    def from_defaults(
        cls,
        service_context: Optional[ServiceContext] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> "SimpleChatEngine":
        """Initialize a SimpleChatEngine from default parameters."""
        service_context = service_context or ServiceContext.from_defaults()
        if not isinstance(service_context.llm_predictor, LLMPredictor):
            raise ValueError("llm_predictor must be a LLMPredictor instance")
        llm = service_context.llm_predictor.llm

        chat_history = chat_history or []
        return cls(llm=llm, chat_history=chat_history)

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))

        chat_response = self._llm.chat(chat_history)
        ai_message = chat_response.message
        chat_history.append(ai_message)

        return Response(response=chat_response.message.content)

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))

        chat_response = StreamingChatResponse(self._llm.stream_chat(chat_history))
        thread = Thread(
            target=chat_response.write_response_to_history, args=(chat_history,)
        )
        thread.start()

        return StreamingResponse(response_gen=chat_response.response_gen)

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:

        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))

        chat_response = await self._llm.achat(chat_history)
        ai_message = chat_response.message
        chat_history.append(ai_message)

        return Response(response=chat_response.message.content)

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        raise NotImplementedError(
            "astream_chat() not implemented for SimpleChatEngine."
        )

    def reset(self) -> None:
        self._chat_history = []

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history as human and ai message pairs."""
        return self._chat_history
