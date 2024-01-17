from typing import Any, Optional, Sequence, Union

from llama_index.bridge.pydantic import Field
from llama_index.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.llms.generic_utils import (
    async_stream_completion_response_to_chat_response,
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.llms.openai import OpenAI, Tokenizer
from llama_index.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)


class OpenAILike(OpenAI):
    """
    OpenAILike is a thin wrapper around the OpenAI model that makes it compatible with
    3rd party tools that provide an openai-compatible api.

    Currently, llama_index prevents using custom models with their OpenAI class
    because they need to be able to infer some metadata from the model name.

    NOTE: You still need to set the OPENAI_BASE_API and OPENAI_API_KEY environment
    variables or the api_key and api_base constructor arguments.
    OPENAI_API_KEY/api_key can normally be set to anything in this case,
    but will depend on the tool you're using.
    """

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=LLMMetadata.__fields__["context_window"].field_info.description,
    )
    is_chat_model: bool = Field(
        default=False,
        description=LLMMetadata.__fields__["is_chat_model"].field_info.description,
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=LLMMetadata.__fields__[
            "is_function_calling_model"
        ].field_info.description,
    )
    tokenizer: Union[Tokenizer, str, None] = Field(
        default=None,
        description=(
            "An instance of a tokenizer object that has an encode method, or the name"
            " of a tokenizer model from Hugging Face. If left as None, then this"
            " disables inference of max_tokens."
        ),
    )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    @property
    def _tokenizer(self) -> Optional[Tokenizer]:
        if isinstance(self.tokenizer, str):
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Please install transformers (pip install transformers) to use "
                    "huggingface tokenizers with OpenAILike."
                ) from exc

            return AutoTokenizer.from_pretrained(self.tokenizer)
        return self.tokenizer

    @classmethod
    def class_name(cls) -> str:
        return "OpenAILike"

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt."""
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        return super().complete(prompt, **kwargs)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Stream complete the prompt."""
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        return super().stream_complete(prompt, **kwargs)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat with the model."""
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = self.complete(prompt, formatted=True, **kwargs)
            return completion_response_to_chat_response(completion_response)

        return super().chat(messages, **kwargs)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
            return stream_completion_response_to_chat_response(completion_response)

        return super().stream_chat(messages, **kwargs)

    # -- Async methods --

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt."""
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        return await super().acomplete(prompt, **kwargs)

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Stream complete the prompt."""
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        return await super().astream_complete(prompt, **kwargs)

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Chat with the model."""
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = await self.acomplete(prompt, formatted=True, **kwargs)
            return completion_response_to_chat_response(completion_response)

        return await super().achat(messages, **kwargs)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = await self.astream_complete(
                prompt, formatted=True, **kwargs
            )
            return async_stream_completion_response_to_chat_response(
                completion_response
            )

        return await super().astream_chat(messages, **kwargs)
