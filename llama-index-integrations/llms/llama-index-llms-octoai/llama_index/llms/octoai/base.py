from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
)

from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
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
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

from llama_index.llms.octoai.utils import (
    octoai_modelname_to_contextsize,
    to_octoai_messages,
)

from octoai.client import OctoAI

DEFAULT_OCTOAI_MODEL = "mistral-7b-instruct"


class OctoAI(LLM):
    model: str = Field(
        default=DEFAULT_OCTOAI_MODEL, description="The model to use with OctoAI"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    timeout: float = Field(
        default=120, description="The timeout to use in seconds.", ge=0
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OctoAI SDK."
    )
    _client: Optional[OctoAI] = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_OCTOAI_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        timeout: int = 120,
        token: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        # base class
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            additional_kwargs=additional_kwargs,
            max_tokens=max_tokens,
            model=model,
            callback_manager=callback_manager,
            temperature=temperature,
            timeout=timeout,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        token = get_from_param_or_env("token", token, "OCTOAI_TOKEN", "")

        if not token:
            raise ValueError(
                "You must provide an API token to use OctoAI. "
                "You can either pass it in as an argument or set it `OCTOAI_TOKEN`."
                "To generate a token in your OctoAI account settings: https://octoai.cloud/settings`."
            )

        try:
            self._client = OctoAI(token=token, timeout=timeout)
        except ImportError as err:
            raise ImportError(
                "Could not import OctoAI python package. "
                "Please install it with `pip install octoai-sdk`."
            ) from err

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=octoai_modelname_to_contextsize(self.model),
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            model_name=self.model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
        response = self._client.chat.completions.create(
            messages=to_octoai_messages(messages), **self._get_all_kwargs(**kwargs)
        )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=response.choices[0].message.content
            ),
            raw=dict(response),
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        streaming_response = self._client.chat.completions.create(
            messages=to_octoai_messages(messages),
            stream=True,
            **self._get_all_kwargs(**kwargs),
        )

        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for completion in streaming_response:
                content_delta = completion.choices[0].delta.content
                if content_delta is None:
                    continue
                content += content_delta

                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=completion,
                )

        return gen()

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    # ===== Async Endpoints =====
    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        raise NotImplementedError

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError
