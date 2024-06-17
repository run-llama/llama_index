from typing import Any, Dict, Optional
import json

# Import LlamaIndex dependencies
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)
from llama_index.core.llms.callbacks import llm_completion_callback, CallbackManager
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.bridge.pydantic import PrivateAttr

from cleanlab_studio import Studio


class CleanlabTLM(CustomLLM):
    # TODO: figure context_window from the underlying model (GPT-3.5 has 16k, GPT-4 has 128k)
    context_window: int = 16000
    max_tokens: int = 512
    model: str = "TLM"

    _client = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        quality_preset: Optional[str] = "medium",
        model: Optional[str] = "TLM",
        max_tokens: Optional[int] = 512,
        callback_manager: Optional[CallbackManager] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize params."""
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager,
        )
        api_key = get_from_param_or_env("api_key", api_key, "CLEANLAB_API_KEY")

        studio = Studio(api_key=api_key)
        self._client = studio.TLM(quality_preset=quality_preset)

    @classmethod
    def class_name(cls) -> str:
        return "CleanlabTLM"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=context_window,
            num_output=max_tokens,
            model_name=model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Prompt TLM for a response and trustworthiness score
        response: Dict[str, str] = self._client.prompt(prompt)
        # output = json.dumps(response)
        return CompletionResponse(
            text=response["response"],
            additional_kwargs={
                "trustworthiness_score": response["trustworthiness_score"]
            },
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Prompt TLM for a response and trustworthiness score
        response = self._client.prompt(prompt)
        output = json.dumps(response)

        # TODO: figure how to stream additional_kwargs. workaround: dump `trustworthiness_score` as str
        # Stream the output
        output_str = ""
        for token in output:
            output_str += token
            yield CompletionResponse(text=output_str, delta=token)
