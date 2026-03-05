from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.core.base.llms.types import ChatMessage, LLMMetadata
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.crusoe.utils import (
    ALL_AVAILABLE_MODELS,
    DEFAULT_CRUSOE_API_BASE,
    crusoe_modelname_to_contextsize,
    is_function_calling_model,
)
from llama_index.llms.openai import OpenAI

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_CONTEXT_WINDOW = 131072


class Crusoe(OpenAI):
    """
    Crusoe LLM.

    Crusoe provides an OpenAI-compatible inference API with low-latency access
    to leading open models, powered by MemoryAlloy cluster-wide KV caching.

    Examples:
        `pip install llama-index-llms-crusoe`

        ```python
        from llama_index.llms.crusoe import Crusoe

        llm = Crusoe(
            model="meta-llama/Llama-3.3-70B-Instruct",
            api_key="YOUR_CRUSOE_API_KEY",
        )
        resp = llm.complete("Paul Graham is ")
        print(resp)
        ```

    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        api_base: Optional[str] = DEFAULT_CRUSOE_API_BASE,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        context_window: Optional[int] = None,
        is_function_calling: Optional[bool] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_base = get_from_param_or_env(
            "api_base", api_base, "CRUSOE_API_BASE", DEFAULT_CRUSOE_API_BASE
        )
        api_key = get_from_param_or_env("api_key", api_key, "CRUSOE_API_KEY")

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
            default_headers=default_headers,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._custom_context_window: Optional[int] = context_window
        self._custom_is_function_calling: Optional[bool] = is_function_calling

    @classmethod
    def class_name(cls) -> str:
        return "Crusoe_LLM"

    def _get_context_window(self) -> int:
        if self._custom_context_window is not None:
            return self._custom_context_window

        try:
            return crusoe_modelname_to_contextsize(self.model)
        except ValueError:
            return DEFAULT_CONTEXT_WINDOW

    def _get_is_function_calling(self) -> bool:
        if self._custom_is_function_calling is not None:
            return self._custom_is_function_calling

        return is_function_calling_model(self.model)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._get_context_window(),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            is_function_calling_model=self._get_is_function_calling(),
        )

    @property
    def _is_chat_model(self) -> bool:
        return True

    @classmethod
    def available_models(cls) -> Dict[str, int]:
        """Returns a mapping of model names to their context window sizes."""
        return ALL_AVAILABLE_MODELS
