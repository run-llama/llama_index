from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_TEMPERATURE

# from mistralai.models.chat_completion import ChatMessage
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
from llama_index.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    get_from_param_or_env,
    stream_chat_to_completion_decorator,
)
from llama_index.llms.llm import LLM
from llama_index.llms.mistralai_utils import (
    mistralai_modelname_to_contextsize,
)
from llama_index.types import BaseOutputParser, PydanticProgramMode

DEFAULT_MISTRALAI_MODEL = "mistral-tiny"
DEFAULT_MISTRALAI_ENDPOINT = "https://api.mistral.ai"
DEFAULT_MISTRALAI_MAX_TOKENS = 512


class MistralAI(LLM):
    model: str = Field(
        default=DEFAULT_MISTRALAI_MODEL, description="The mistralai model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_MISTRALAI_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    timeout: float = Field(
        default=120, description="The timeout to use in seconds.", gte=0
    )
    max_retries: int = Field(
        default=5, description="The maximum number of API retries.", gte=0
    )
    safe_mode: bool = Field(
        default=False,
        description="The parameter to enforce guardrails in chat generations.",
    )
    random_seed: str = Field(
        default=None, description="The random seed to use for sampling."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the MistralAI API."
    )

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_MISTRALAI_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MISTRALAI_MAX_TOKENS,
        timeout: int = 120,
        max_retries: int = 5,
        safe_mode: bool = False,
        random_seed: Optional[int] = None,
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
            from mistralai.async_client import MistralAsyncClient
            from mistralai.client import MistralClient
        except ImportError as e:
            raise ImportError(
                "You must install the `mistralai` package to use mistralai."
                "Please `pip install mistralai`"
            ) from e

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_key = get_from_param_or_env("api_key", api_key, "MISTRAL_API_KEY", "")

        if not api_key:
            raise ValueError(
                "You must provide an API key to use mistralai. "
                "You can either pass it in as an argument or set it `MISTRAL_API_KEY`."
            )

        self._client = MistralClient(
            api_key=api_key,
            endpoint=DEFAULT_MISTRALAI_ENDPOINT,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._aclient = MistralAsyncClient(
            api_key=api_key,
            endpoint=DEFAULT_MISTRALAI_ENDPOINT,
            timeout=timeout,
            max_retries=max_retries,
        )

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            safe_mode=safe_mode,
            random_seed=random_seed,
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
        return "MistralAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=mistralai_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            safe_mode=self.safe_mode,
            random_seed=self.random_seed,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "random_seed": self.random_seed,
            "safe_mode": self.safe_mode,
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
        # convert messages to mistral ChatMessage
        from mistralai.client import ChatMessage as mistral_chatmessage

        messages = [
            mistral_chatmessage(role=x.role, content=x.content) for x in messages
        ]
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = self._client.chat(messages=messages, **all_kwargs)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=response.choices[0].message.content
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
        # convert messages to mistral ChatMessage
        from mistralai.client import ChatMessage as mistral_chatmessage

        messages = [
            mistral_chatmessage(role=message.role, content=message.content)
            for message in messages
        ]
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.chat_stream(messages=messages, **all_kwargs)

        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for chunk in response:
                content_delta = chunk.choices[0].delta.content
                if content_delta is None:
                    continue
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=chunk,
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
        # convert messages to mistral ChatMessage
        from mistralai.client import ChatMessage as mistral_chatmessage

        messages = [
            mistral_chatmessage(role=message.role, content=message.content)
            for message in messages
        ]
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = await self._aclient.chat(messages=messages, **all_kwargs)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=response.choices[0].message.content
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
        # convert messages to mistral ChatMessage
        from mistralai.client import ChatMessage as mistral_chatmessage

        messages = [
            mistral_chatmessage(role=x.role, content=x.content) for x in messages
        ]
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._aclient.chat_stream(messages=messages, **all_kwargs)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            role = MessageRole.ASSISTANT
            async for chunk in response:
                content_delta = chunk.choices[0].delta.content
                if content_delta is None:
                    continue
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=chunk,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)
