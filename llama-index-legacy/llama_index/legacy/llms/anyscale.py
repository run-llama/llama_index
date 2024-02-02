from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.llms.types import ChatMessage, LLMMetadata
from llama_index.llms.anyscale_utils import (
    anyscale_modelname_to_contextsize,
)
from llama_index.llms.generic_utils import get_from_param_or_env
from llama_index.llms.openai import OpenAI
from llama_index.types import BaseOutputParser, PydanticProgramMode

DEFAULT_API_BASE = "https://api.endpoints.anyscale.com/v1"
DEFAULT_MODEL = "meta-llama/Llama-2-70b-chat-hf"


class Anyscale(OpenAI):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        api_base: Optional[str] = DEFAULT_API_BASE,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_base = get_from_param_or_env("api_base", api_base, "ANYSCALE_API_BASE")
        api_key = get_from_param_or_env("api_key", api_key, "ANYSCALE_API_KEY")

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Anyscale_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=anyscale_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
        )

    @property
    def _is_chat_model(self) -> bool:
        return True
