import os
from pydantic import Field, PrivateAttr
from typing import Any, Optional

from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.llms.base import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM


class PredibaseLLM(CustomLLM):
    """Predibase LLM"""

    model_name: str = Field(description="The Predibase model to use.")
    predibase_api_key: str = Field(description="The Predibase API key to use.")
    max_new_tokens: int = Field(description="The number of tokens to generate.")
    temperature: float = Field(description="The temperature to use for sampling.")
    context_window: int = Field(
        description="The number of context tokens available to the LLM."
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        predibase_api_key: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
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
        )

    def initialize_client(self, predibase_api_key: str) -> Any:
        try:
            from predibase import PredibaseClient

            pc = PredibaseClient(token=self.predibase_api_key)
            return pc
        except ImportError as e:
            raise ImportError(
                "Could not import Predibase Python package. "
                "Please install it with `pip install predibase`."
            ) from e
        except ValueError as e:
            raise ValueError("Your API key is not correct. Please try again") from e

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
        )

    # "CompletionResponse"
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> "CompletionResponse":
        model_kwargs = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }
        results = self._client.prompt(prompt, self.model_name, options=model_kwargs)
        return CompletionResponse(
            text=results.loc[0, "response"], additional_kwargs=model_kwargs
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> "CompletionResponseGen":
        raise NotImplementedError()
