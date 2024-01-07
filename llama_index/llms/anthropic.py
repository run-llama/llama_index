from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.types import (
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
from llama_index.llms.anthropic_utils import (
    anthropic_modelname_to_contextsize,
    messages_to_anthropic_prompt,
)
from llama_index.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.llms.llm import LLM
from llama_index.types import BaseOutputParser, PydanticProgramMode

DEFAULT_ANTHROPIC_MODEL = "claude-2"
DEFAULT_ANTHROPIC_MAX_TOKENS = 512


class Anthropic(LLM):
    model: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL, description="The anthropic model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_ANTHROPIC_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    base_url: Optional[str] = Field(default=None, description="The base URL to use.")
    timeout: Optional[float] = Field(
        default=None, description="The timeout to use in seconds.", gte=0
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", gte=0
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the anthropic API."
    )

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "You must install the `anthropic` package to use Anthropic."
                "Please `pip install anthropic`"
            ) from e

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        self._client = anthropic.Anthropic(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        self._aclient = anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            model=model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Anthropic_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=anthropic_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens_to_sample": self.max_tokens,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = messages_to_anthropic_prompt(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.completions.create(
            prompt=prompt, stream=False, **all_kwargs
        )
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=response.completion
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = messages_to_anthropic_prompt(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.completions.create(
            prompt=prompt, stream=True, **all_kwargs
        )

        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for r in response:
                content_delta = r.completion
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=r,
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        prompt = messages_to_anthropic_prompt(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._aclient.completions.create(
            prompt=prompt, stream=False, **all_kwargs
        )
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=response.completion
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        prompt = messages_to_anthropic_prompt(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._aclient.completions.create(
            prompt=prompt, stream=True, **all_kwargs
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            role = MessageRole.ASSISTANT
            async for r in response:
                content_delta = r.completion
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=r,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)
