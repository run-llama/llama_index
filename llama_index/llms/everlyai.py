from typing import Any, Dict, Optional

from llama_index.callbacks import CallbackManager
from llama_index.llms.base import LLMMetadata
from llama_index.llms.everlyai_utils import everlyai_modelname_to_contextsize
from llama_index.llms.generic_utils import get_from_param_or_env
from llama_index.llms.openai import OpenAI

EVERLYAI_API_BASE = "https://everlyai.xyz/hosted"
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"


class EverlyAI(OpenAI):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 256,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_key = get_from_param_or_env("api_key", api_key, "EverlyAI_API_KEY")

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=EVERLYAI_API_BASE,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "EverlyAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=everlyai_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
        )

    @property
    def _is_chat_model(self) -> bool:
        return True
