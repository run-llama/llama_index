from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Awaitable,
    List,
    Literal,
)

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
from llama_index.core.bridge.pydantic import Field, PrivateAttr, BaseModel
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    get_from_param_or_env,
    stream_chat_to_completion_decorator,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

from llama_index.llms.nvidia.utils import (
    playground_modelname_to_contextsize,
    API_CATALOG_MODELS,
)

from llama_index.llms.openai.utils import (
    from_openai_message,
    to_openai_message_dicts,
)

from openai import OpenAI as SyncOpenAI
from openai import AsyncOpenAI

DEFAULT_PLAYGROUND_MODEL = "mistralai/mistral-7b-instruct-v0.2"
BASE_PLAYGROUND_URL = "https://integrate.api.nvidia.com/v1/"
DEFAULT_PLAYGROUND_MAX_TOKENS = 512


class Model(BaseModel):
    id: str


class NVIDIA(LLM):
    """NVIDIA's API Catalog Connector."""

    model: str = Field(
        default=DEFAULT_PLAYGROUND_MODEL,
        description="The NVIDIA API Catalog model to use.",
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_PLAYGROUND_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gte=0,
    )

    timeout: float = Field(
        default=120, description="The timeout for the API request in seconds.", gte=0
    )

    max_retries: int = Field(
        default=5,
        description="The maximum number of retries for the API request.",
        gte=0,
    )

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()
    _mode: str = PrivateAttr("nvidia")

    def __init__(
        self,
        model: str = DEFAULT_PLAYGROUND_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_PLAYGROUND_MAX_TOKENS,
        timeout: float = 120,
        max_retries: int = 5,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        complettion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        callback_manager = callback_manager or CallbackManager([])

        api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        self._client = SyncOpenAI(
            api_key=api_key,
            base_url=BASE_PLAYGROUND_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._client._custom_headers = {"User-Agent": "llama-index-llms-nvidia"}
        self._aclient = AsyncOpenAI(
            api_key=api_key,
            base_url=BASE_PLAYGROUND_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._aclient._custom_headers = {"User-Agent": "llama-index-llms-nvidia"}

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            complettion_to_prompt=complettion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @property
    def available_models(self) -> List[Model]:
        ids = API_CATALOG_MODELS.keys()
        if self._mode == "nim":
            ids = [model.id for model in self._client.models.list()]
        return [Model(id=name) for name in ids]

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIA"

    @property
    def metadata(self) -> LLMMetadata:
        params = {
            "num_output": self.max_tokens,
            "is_chat_model": True,
            "model_name": self.model,
        }
        if context_window := playground_modelname_to_contextsize(self.model):
            params["context_window"] = context_window
        return LLMMetadata(**params)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def mode(
        self,
        mode: Optional[Literal["nvidia", "nim"]] = "nvidia",
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "NVIDIA":
        """
        Change the mode.

        There are two modes, "nvidia" and "nim". The "nvidia" mode is the default
        mode and is used to interact with hosted NIMs. The "nim" mode is used to
        interact with NVIDIA NIM endpoints, which are typically hosted on-premises.

        For the "nvidia" mode, the "api_key" parameter is available to specify
        your API key. If not specified, the NVIDIA_API_KEY environment variable
        will be used.

        For the "nim" mode, the "base_url" parameter is required and the "model"
        parameter may be necessary. Set base_url to the url of your local NIM
        endpoint. For instance, "https://localhost:9999/v1". Additionally, the
        "model" parameter must be set to the name of the model inside the NIM.
        """
        # if mode == "nvidia":
        #     if not api_key:
        #         warnings.warn(
        #             "'nvidia' mode without an api_key may result in an error",
        #             UserWarning,
        #         )
        if mode == "nim":
            if not base_url:
                raise ValueError("base_url is required for nim mode")

        self._mode = mode
        if base_url:
            self._client.base_url = base_url
            self._aclient.base_url = base_url
        if model:
            self.model = model
            self._client.model = model
            self._aclient.model = model
        if api_key:
            self._client.api_key = api_key
            self._aclient.api_key = api_key

        return self

    # === Helper Methods ===

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def _get_response_token_counts(self, raw_response: Any) -> dict:
        """Get the token usage reported by the response."""
        if not isinstance(raw_response, dict):
            return {}

        usage = raw_response.get("usage", {})
        if usage is None:
            return {}

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    # === Sync Methods ===

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        message_dicts = to_openai_message_dicts(messages)

        response = self._client.chat.completions.create(
            messages=message_dicts,
            stream=False,
            **self._get_all_kwargs(**kwargs),
        )
        playground_openai_message = response.choices[0].message
        message = from_openai_message(playground_openai_message)

        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
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
        all_kwargs = self._get_all_kwargs(**kwargs)
        message_dicts = to_openai_message_dicts(messages)

        response = self._client.chat.completions.create(
            messages=message_dicts, stream=True, **all_kwargs
        )

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

    # === Async Methods ===

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        achat_fn: Callable[..., Awaitable[ChatResponse]]
        achat_fn = self._achat
        return await achat_fn(messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        message_dicts = to_openai_message_dicts(messages)

        response = await self._aclient.chat.completions.create(
            messages=message_dicts, stream=True, **all_kwargs
        )

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
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self._achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        message_dicts = to_openai_message_dicts(messages)
        response = await self._aclient.chat.completions.create(
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        message_dict = response.choices[0].message
        message = from_openai_message(message_dict)

        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )
