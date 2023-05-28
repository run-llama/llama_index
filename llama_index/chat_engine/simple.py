from typing import Any, List, Optional, Tuple

from llama_index.chat_engine.types import BaseChatEngine, ChatHistoryType
from llama_index.chat_engine.utils import get_chat_history
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.base import Prompt
from llama_index.response.schema import RESPONSE_TYPE, Response

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

DEFAULT_PROMPT = Prompt(DEFAULT_TMPL)


class SimpleChatEngine(BaseChatEngine):
    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        prompt: Optional[Prompt] = None,
        chat_history: Optional[ChatHistoryType] = None,
    ) -> None:
        self._service_context = service_context or ServiceContext.from_defaults()
        self._prompt = prompt or DEFAULT_PROMPT
        self._chat_history = chat_history or []

    @classmethod
    def from_defaults(
        cls,
        service_context: Optional[ServiceContext] = None,
        prompt: Optional[Prompt] = None,
        chat_history: Optional[ChatHistoryType] = None,
        **kwargs: Any,
    ) -> "SimpleChatEngine":
        return cls(
            service_context=service_context, prompt=prompt, chat_history=chat_history
        )

    def chat(self, message: str) -> RESPONSE_TYPE:
        history = get_chat_history(self._chat_history)
        response, _ = self._service_context.llm_predictor.predict(
            self._prompt,
            history=history,
            message=message,
        )

        # Record response
        self._chat_history.append((message, str(response)))

        return Response(response=response)

    async def achat(self, message: str) -> RESPONSE_TYPE:
        history = get_chat_history(self._chat_history)
        response, _ = await self._service_context.llm_predictor.apredict(
            self._prompt,
            history=history,
            message=message,
        )

        # Record response
        self._chat_history.append((message, str(response)))

        return response

    def reset(self) -> None:
        self._chat_history = []
