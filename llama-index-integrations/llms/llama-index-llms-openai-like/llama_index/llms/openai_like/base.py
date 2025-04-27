from typing import Any, Optional, Sequence, Union

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
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.base.llms.generic_utils import (
    async_stream_completion_response_to_chat_response,
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.llms.openai.base import OpenAI, Tokenizer
from transformers import AutoTokenizer


class OpenAILike(OpenAI):
    """OpenaAILike LLM.

    OpenAILike is a thin wrapper around the OpenAI model that makes it compatible with
    3rd party tools that provide an openai-compatible api.

    Args:
        model (str):
            The model to use for the api.
        api_base (str):
            The base url to use for the api.
            Defaults to "https://api.openai.com/v1".
        is_chat_model (bool):
            Whether the model uses the chat or completion endpoint.
            Defaults to False.
        is_function_calling_model (bool):
            Whether the model supports OpenAI function calling/tools over the API.
            Defaults to False.
        api_key (str):
            The api key to use for the api.
            Set this to some random string if your API does not require an api key.
        context_window (int):
            The context window to use for the api. Set this to your model's context window for the best experience.
            Defaults to 3900.
        max_tokens (int):
            The max number of tokens to generate.
            Defaults to None.
        temperature (float):
            The temperature to use for the api.
            Default is 0.1.
        additional_kwargs (dict):
            Specify additional parameters to the request body.
        max_retries (int):
            How many times to retry the API call if it fails.
            Defaults to 3.
        timeout (float):
            How long to wait, in seconds, for an API call before failing.
            Defaults to 60.0.
        reuse_client (bool):
            Reuse the OpenAI client between requests.
            Defaults to True.
        default_headers (dict):
            Override the default headers for API requests.
            Defaults to None.
        http_client (httpx.Client):
            Pass in your own httpx.Client instance.
            Defaults to None.
        async_http_client (httpx.AsyncClient):
            Pass in your own httpx.AsyncClient instance.
            Defaults to None.

    Examples:
        `pip install llama-index-llms-openai-like`

        ```python
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="my model",
            api_base="https://hostname.com/v1",
            api_key="fake",
            context_window=128000,
            is_chat_model=True,
            is_function_calling_model=False,
        )

        response = llm.complete("Hello World!")
        print(str(response))
        ```
    """

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=LLMMetadata.model_fields["context_window"].description,
    )
    is_chat_model: bool = Field(
        default=False,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=LLMMetadata.model_fields["is_function_calling_model"].description,
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
