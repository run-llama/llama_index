from typing import Any, Callable, Dict, Optional, Sequence

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
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.friendli.utils import (
    friendli_modelname_to_contextsize,
    get_chat_request,
)

import friendli


class Friendli(LLM):
    """Friendli LLM."""

    model: str = Field(description="The friendli model to use.")
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    temperature: Optional[float] = Field(
        description="The temperature to use for sampling."
    )

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Friendli API."
    )

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "mixtral-8x7b-instruct-v0-1",
        friendli_token: Optional[str] = None,
        max_tokens: int = 256,
        temperature: Optional[float] = 0.1,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._client = friendli.Friendli(token=friendli_token)
        self._aclient = friendli.AsyncFriendli(token=friendli_token)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Friendli_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=friendli_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.chat.completions.create(
            stream=False,
            **get_chat_request(messages),
            **all_kwargs,
        )
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=response.choices[0].message.content
            ),
            raw=response.__dict__,
            additional_kwargs={"usage": response.usage.__dict__},
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.completions.create(
            prompt=prompt,
            stream=False,
            **all_kwargs,
        )
        return CompletionResponse(
            text=response.choices[0].text,
            additional_kwargs={"usage": response.usage.__dict__},
            raw=response.__dict__,
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)

        stream = self._client.chat.completions.create(
            stream=True,
            **get_chat_request(messages),
            **all_kwargs,
        )

        def gen() -> ChatResponseGen:
            content = ""
            for chunk in stream:
                content_delta = chunk.choices[0].delta.content or ""
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                    delta=content_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)

        stream = self._client.completions.create(
            prompt=prompt,
            stream=True,
            **all_kwargs,
        )

        def gen() -> CompletionResponseGen:
            content = ""
            for chunk in stream:
                content_delta = chunk.text
                content += content_delta
                yield CompletionResponse(
                    text=content,
                    delta=content_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._aclient.chat.completions.create(
            stream=False,
            **get_chat_request(messages),
            **all_kwargs,
        )
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=response.choices[0].message.content
            ),
            raw=response.__dict__,
            additional_kwargs={"usage": response.usage.__dict__},
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._aclient.completions.create(
            prompt=prompt,
            stream=False,
            **all_kwargs,
        )
        return CompletionResponse(
            text=response.choices[0].text,
            additional_kwargs={"usage": response.usage.__dict__},
            raw=response.__dict__,
        )

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        all_kwargs = self._get_all_kwargs(**kwargs)

        stream = await self._aclient.chat.completions.create(
            stream=True,
            **get_chat_request(messages),
            **all_kwargs,
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            async for chunk in stream:
                content_delta = chunk.choices[0].delta.content or ""
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                    delta=content_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        all_kwargs = self._get_all_kwargs(**kwargs)

        stream = await self._aclient.completions.create(
            prompt=prompt,
            stream=True,
            **all_kwargs,
        )

        async def gen() -> CompletionResponseAsyncGen:
            content = ""
            async for chunk in stream:
                content_delta = chunk.text
                content += content_delta
                yield CompletionResponse(
                    text=content,
                    delta=content_delta,
                    raw=chunk.__dict__,
                )

        return gen()
