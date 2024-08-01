from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.legacy.llms.ai21_utils import ai21_model_to_context_size
from llama_index.legacy.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.legacy.llms.custom import CustomLLM
from llama_index.legacy.llms.generic_utils import (
    completion_to_chat_decorator,
    get_from_param_or_env,
)
from llama_index.legacy.types import BaseOutputParser, PydanticProgramMode


class AI21(CustomLLM):
    """AI21 Labs LLM."""

    model: str = Field(description="The AI21 model to use.")
    maxTokens: int = Field(description="The maximum number of tokens to generate.")
    temperature: float = Field(description="The temperature to use for sampling.")

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the anthropic API."
    )

    _api_key = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = "j2-mid",
        maxTokens: Optional[int] = 512,
        temperature: Optional[float] = 0.1,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """Initialize params."""
        try:
            import ai21 as _  # noqa
        except ImportError as e:
            raise ImportError(
                "You must install the `ai21` package to use AI21."
                "Please `pip install ai21`"
            ) from e

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_key = get_from_param_or_env("api_key", api_key, "AI21_API_KEY")
        self._api_key = api_key

        super().__init__(
            model=model,
            maxTokens=maxTokens,
            temperature=temperature,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(self) -> str:
        """Get Class Name."""
        return "AI21_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=ai21_model_to_context_size(self.model),
            num_output=self.maxTokens,
            model_name=self.model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "maxTokens": self.maxTokens,
            "temperature": self.temperature,
        }
        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        import ai21

        ai21.api_key = self._api_key

        response = ai21.Completion.execute(**all_kwargs, prompt=prompt)

        return CompletionResponse(
            text=response["completions"][0]["data"]["text"], raw=response.__dict__
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError(
            "AI21 does not currently support streaming completion."
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_fn = completion_to_chat_decorator(self.complete)

        return chat_fn(messages, **all_kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError("AI21 does not Currently Support Streaming Chat.")
