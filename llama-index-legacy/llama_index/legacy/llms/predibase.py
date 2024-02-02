import os
from typing import Any, Callable, Optional, Sequence

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.legacy.llms.base import llm_completion_callback
from llama_index.legacy.llms.custom import CustomLLM
from llama_index.legacy.types import BaseOutputParser, PydanticProgramMode


class PredibaseLLM(CustomLLM):
    """Predibase LLM."""

    model_name: str = Field(description="The Predibase model to use.")
    predibase_api_key: str = Field(description="The Predibase API key to use.")
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The number of context tokens available to the LLM.",
        gt=0,
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        predibase_api_key: Optional[str] = None,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        temperature: float = DEFAULT_TEMPERATURE,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        predibase_api_key = (
            predibase_api_key
            if predibase_api_key
            else os.environ.get("PREDIBASE_API_TOKEN")
        )
        assert predibase_api_key is not None

        self._client = self.initialize_client(predibase_api_key)

        super().__init__(
            model_name=model_name,
            predibase_api_key=predibase_api_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            context_window=context_window,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @staticmethod
    def initialize_client(predibase_api_key: str) -> Any:
        try:
            from predibase import PredibaseClient

            return PredibaseClient(token=predibase_api_key)
        except ImportError as e:
            raise ImportError(
                "Could not import Predibase Python package. "
                "Please install it with `pip install predibase`."
            ) from e
        except ValueError as e:
            raise ValueError("Your API key is not correct. Please try again") from e

    @classmethod
    def class_name(cls) -> str:
        return "PredibaseLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> "CompletionResponse":
        llm = self._client.LLM(f"pb://deployments/{self.model_name}")
        results = llm.prompt(
            prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature
        )
        return CompletionResponse(text=results.response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> "CompletionResponseGen":
        raise NotImplementedError
