from typing import Any, Optional

from llama_index.chat_engine.types import BaseChatEngine, ChatHistoryType
from llama_index.chat_engine.utils import (
    is_chat_model,
    to_chat_buffer,
    to_langchain_chat_history,
)
from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType
from llama_index.response.schema import RESPONSE_TYPE, Response
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatGeneration

DEFAULT_TMPL = """\
Assistant is a versatile language model that can assist with various tasks. \
It can answer questions, provide detailed explanations, and engage in discussions on \
diverse subjects. By processing and understanding extensive text data, Assistant \
offers coherent and relevant responses. It continually learns and improves, expanding \
its capabilities. With the ability to generate its own text, Assistant can facilitate \
conversations, offer explanations, and describe a broad range of topics. Whether you \
require assistance with a specific inquiry or desire a conversation on a particular \
subject, Assistant is a valuable resource at your disposal.

{history}
Human: {message}
Assistant: 
"""

DEFAULT_PROMPT = Prompt(DEFAULT_TMPL, prompt_type=PromptType.CONVERSATION)


class SimpleChatEngine(BaseChatEngine):
    """Simple Chat Engine.

    Have a conversation with the LLM.
    This does not make use of a knowledge base.
    """

    def __init__(
        self,
        service_context: ServiceContext,
        prompt: Prompt,
        chat_history: ChatHistoryType,
    ) -> None:
        self._service_context = service_context
        self._prompt = prompt
        self._chat_history = chat_history

    @classmethod
    def from_defaults(
        cls,
        service_context: Optional[ServiceContext] = None,
        prompt: Optional[Prompt] = None,
        chat_history: Optional[ChatHistoryType] = None,
        **kwargs: Any,
    ) -> "SimpleChatEngine":
        """Initialize a SimpleChatEngine from default parameters."""
        service_context = service_context or ServiceContext.from_defaults()
        prompt = prompt or DEFAULT_PROMPT
        chat_history = chat_history or []
        return cls(
            service_context=service_context, prompt=prompt, chat_history=chat_history
        )

    def chat(self, message: str) -> RESPONSE_TYPE:
        if is_chat_model(self._service_context):
            assert isinstance(self._service_context.llm_predictor, LLMPredictor)
            llm = self._service_context.llm_predictor.llm
            assert isinstance(llm, BaseChatModel)
            history = to_langchain_chat_history(self._chat_history)
            history.add_user_message(message=message)
            result = llm.generate([history.messages])
            generation = result.generations[0][0]
            assert isinstance(generation, ChatGeneration)
            response = generation.message.content
        else:
            history_buffer = to_chat_buffer(self._chat_history)
            response, _ = self._service_context.llm_predictor.predict(
                self._prompt,
                history=history_buffer,
                message=message,
            )

        # Record response
        self._chat_history.append((message, str(response)))

        return Response(response=response)

    async def achat(self, message: str) -> RESPONSE_TYPE:
        if is_chat_model(self._service_context):
            assert isinstance(self._service_context.llm_predictor, LLMPredictor)
            llm = self._service_context.llm_predictor.llm
            assert isinstance(llm, BaseChatModel)
            history = to_langchain_chat_history(self._chat_history)
            history.add_user_message(message=message)
            result = await llm.agenerate([history.messages])
            generation = result.generations[0][0]
            assert isinstance(generation, ChatGeneration)
            response = generation.message.content
        else:
            history_buffer = to_chat_buffer(self._chat_history)
            response, _ = await self._service_context.llm_predictor.apredict(
                self._prompt,
                history=history_buffer,
                message=message,
            )

        # Record response
        self._chat_history.append((message, str(response)))

        return Response(response=response)

    def reset(self) -> None:
        self._chat_history = []
