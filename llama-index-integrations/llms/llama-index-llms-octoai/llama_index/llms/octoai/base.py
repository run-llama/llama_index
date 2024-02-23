from typing import Optional, List, Mapping, Any, Dict

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
from octoai.chat import TextModel
from octoai.client import Client
from octoai.chat import TextModel

import json

DEFAULT_OCTOAI_MODEL = TextModel.MIXTRAL_8X7B_INSTRUCT_FP16


class OctoAI(CustomLLM):
    model: str = Field(
        default=DEFAULT_OCTOAI_MODEL, description="The model to use with OctoAI"
    )

    _client: Client = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_OCTOAI_MODEL,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        print(f"Hello from OctoAI Integration ... with model {model}")
        self._client = Client()

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            additional_kwargs=additional_kwargs,
            model=model,
            callback_manager=callback_manager,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata()

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
        )
        print(json.dumps(completion.dict(), indent=2))
        return CompletionResponse(text="test")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = ""
        for token in "test response stream":
            response += token
            yield CompletionResponse(text=response, delta=token)
