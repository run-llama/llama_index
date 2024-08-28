import os
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

try:
    from reka.client import Reka, AsyncReka
    from reka.core import ApiError
except ImportError:
    raise ValueError(
        "Reka is not installed. Please install it with `pip install reka-api`."
    )

REKA_MODELS = [
    "reka-edge",
    "reka-flash",
    "reka-core",
    "reka-core-20240501",
]

DEFAULT_REKA_MODEL = "reka-core-20240501"
DEFAULT_REKA_MAX_TOKENS = 512


def process_messages_for_reka(messages: Sequence[ChatMessage]) -> List[Dict[str, str]]:
    reka_messages = []
    system_message = None

    for message in messages:
        if message.role == MessageRole.SYSTEM:
            if system_message is None:
                system_message = message.content
            else:
                raise ValueError("Multiple system messages are not supported.")
        elif message.role == MessageRole.USER:
            content = message.content
            if system_message:
                content = f"{system_message}\n{content}"
                system_message = None
            reka_messages.append({"role": "user", "content": content})
        elif message.role == MessageRole.ASSISTANT:
            reka_messages.append({"role": "assistant", "content": message.content})
        else:
            raise ValueError(f"Unsupported message role: {message.role}")

    return reka_messages


class RekaLLM(CustomLLM):
    """Reka LLM integration for LlamaIndex."""

    model: str = Field(default=DEFAULT_REKA_MODEL, description="The Reka model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_REKA_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for Reka API calls.",
    )

    _client: Reka = PrivateAttr()
    _aclient: AsyncReka = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_REKA_MODEL,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_REKA_MAX_TOKENS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_key = api_key or os.getenv("REKA_API_KEY")
        if not api_key:
            raise ValueError(
                "Reka API key is required. Please provide it as an argument or set the REKA_API_KEY environment variable."
            )

        self._client = Reka(api_key=api_key)
        self._aclient = AsyncReka(api_key=api_key)

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,  # Adjust based on the specific Reka model
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {**self._model_kwargs, **kwargs}

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        try:
            response = self._client.chat.create(messages=reka_messages, **all_kwargs)
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.responses[0].message.content,
                ),
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        try:
            response = self._client.chat.create(
                messages=[{"role": "user", "content": prompt}], **all_kwargs
            )
            return CompletionResponse(
                text=response.responses[0].message.content,
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        try:
            stream = self._client.chat.create_stream(
                messages=reka_messages, **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        def gen() -> ChatResponseGen:
            prev_content = ""
            for chunk in stream:
                content = chunk.responses[0].chunk.content
                content_delta = content[len(prev_content) :]
                prev_content = content
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=content,
                    ),
                    delta=content_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        try:
            stream = self._client.chat.create_stream(
                messages=[{"role": "user", "content": prompt}], **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        def gen() -> CompletionResponseGen:
            prev_text = ""
            for chunk in stream:
                text = chunk.responses[0].chunk.content
                text_delta = text[len(prev_text) :]
                prev_text = text
                yield CompletionResponse(
                    text=text,
                    delta=text_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        try:
            response = await self._aclient.chat.create(
                messages=reka_messages, **all_kwargs
            )
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.responses[0].message.content,
                ),
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        try:
            response = await self._aclient.chat.create(
                messages=[{"role": "user", "content": prompt}], **all_kwargs
            )
            return CompletionResponse(
                text=response.responses[0].message.content,
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        try:
            stream = self._aclient.chat.create_stream(
                messages=reka_messages, **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        async def gen() -> ChatResponseAsyncGen:
            prev_content = ""
            async for chunk in stream:
                content = chunk.responses[0].chunk.content
                content_delta = content[len(prev_content) :]
                prev_content = content
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=content,
                    ),
                    delta=content_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        try:
            stream = self._aclient.chat.create_stream(
                messages=[{"role": "user", "content": prompt}], **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        async def gen() -> CompletionResponseAsyncGen:
            prev_text = ""
            async for chunk in stream:
                text = chunk.responses[0].chunk.content
                text_delta = text[len(prev_text) :]
                prev_text = text
                yield CompletionResponse(
                    text=text,
                    delta=text_delta,
                    raw=chunk.__dict__,
                )

        return gen()
