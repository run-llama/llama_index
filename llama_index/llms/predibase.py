import os
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

    def __init__(
        self,
        model_name: str,
        predibase_api_key: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self.predibase_api_key = (
            predibase_api_key
            if predibase_api_key
            else os.environ.get("PREDIBASE_API_TOKEN")
        )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.context_window = context_window
        self.client = self.initialize_client()
        self.callback_manager = callback_manager or CallbackManager([])

    def initialize_client(self) -> Any:
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
        results = self.client.prompt(prompt, self.model_name, options=model_kwargs)
        return CompletionResponse(
            text=results.loc[0, "response"], additional_kwargs=model_kwargs
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> "CompletionResponseGen":
        raise NotImplementedError()
