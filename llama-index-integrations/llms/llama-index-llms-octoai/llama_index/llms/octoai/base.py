from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    cast,
    runtime_checkable,
)

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.octoai.utils import (
    octoai_modelname_to_contextsize,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core import Settings
from octoai.chat import TextModel
from octoai.client import Client
from octoai.chat import TextModel

import json

DEFAULT_OCTOAI_MODEL = TextModel.MIXTRAL_8X7B_INSTRUCT_FP16


class OctoAI(LLM):
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
        return LLMMetadata(
            context_window=octoai_modelname_to_contextsize(self._get_model_name()),
            num_output=self.max_tokens or -1,
            model_name=self.model,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise (ValueError("Not Implemented"))

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise (ValueError("Not Implemented"))

    # ===== Async Endpoints =====
    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        raise (ValueError("Not Implemented"))

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise (ValueError("Not Implemented"))

    # @llm_completion_callback()
    # def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
    #     completion = self._client.chat.completions.create(
    #         model=self.model,
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    #             },
    #             {"role": "user", "content": prompt},
    #         ],
    #         max_tokens=150,
    #     )
    #     print(json.dumps(completion.dict(), indent=2))
    #     return CompletionResponse(text="test")

    # @llm_completion_callback()
    # def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
    #     response = ""
    #     for token in "test response stream":
    #         response += token
    #         yield CompletionResponse(text=response, delta=token)

    # def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
    #     base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
    #     if self.max_tokens is not None:
    #         # If max_tokens is None, don't include in the payload:
    #         # https://platform.openai.com/docs/api-reference/chat
    #         # https://platform.openai.com/docs/api-reference/completions
    #         base_kwargs["max_tokens"] = self.max_tokens
    #     return {**base_kwargs, **self.additional_kwargs}
