from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.multi_modal_llms import (
    MultiModalCompletionResponse,
    MultiModalCompletionResponseAsyncGen,
    MultiModalCompletionResponseGen,
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.multi_modal_llms.openai_utils import (
    to_openai_multi_modal_payload,
)
from llama_index.schema import ImageDocument


class OpenAIMultiModal(MultiModalLLM):
    model: str = Field(description="The Multi-Modal model to use from OpenAI GPT4V.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: int = Field(
        description=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt"
    )
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
    prompt_key: str = Field(description="The key to use for the prompt in API calls.")
    image_key: str = Field(description="The key to use for the image in API calls.")
    image_detail: str = Field(
        description="The level of details for image in API calls."
    )

    max_retries: int = Field(
        default=10, description="Maximum number of retries.", gte=0
    )
    api_key: str = Field(default=None, description="The OpenAI API key.", exclude=True)
    api_base: str = Field(description="The base URL for OpenAI API.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()
    _client: SyncOpenAI = PrivateAttr()
    _aclient: AsyncOpenAI = PrivateAttr()

    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        temperature: float = 0.75,
        max_new_tokens: int = 300,
        num_input_files: int = 100,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        prompt_key: str = "text",
        image_key: str = "image_url",
        max_retries: int = 10,
        image_detail: str = "low",
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://api.openai.com/v1",
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)
        api_key = api_key
        api_base = api_base

        super().__init__(
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_input_files=num_input_files,
            additional_kwargs=additional_kwargs or {},
            context_window=context_window,
            prompt_key=prompt_key,
            image_key=image_key,
            image_detail=image_detail,
            max_retries=max_retries,
            api_key=api_key,
            api_base=api_base,
            callback_manager=callback_manager,
        )
        self._client, self._aclient = self._get_clients(**kwargs)

    def _get_clients(self, **kwargs: Any) -> Tuple[SyncOpenAI, AsyncOpenAI]:
        client = SyncOpenAI(**self._get_credential_kwargs())
        aclient = AsyncOpenAI(**self._get_credential_kwargs())
        return client, aclient

    @classmethod
    def class_name(cls) -> str:
        return "openai_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    def _get_credential_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            **kwargs,
        }

    def _get_multi_modal_input_dict(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> List[ChatCompletionMessageParam]:
        return to_openai_multi_modal_payload(
            prompt=prompt,
            image_documents=image_documents,
            image_detail=self.image_detail,
        )

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        base_kwargs = {"model": self.model, **kwargs}
        if self.max_new_tokens is not None:
            # If max_tokens is None, don't include in the payload:
            # https://platform.openai.com/docs/api-reference/chat
            # https://platform.openai.com/docs/api-reference/completions
            base_kwargs["max_tokens"] = str(self.max_new_tokens)
        return {**base_kwargs, **self.additional_kwargs}

    def _get_response_token_counts(self, raw_response: Any) -> dict:
        """Get the token usage reported by the response."""
        if not isinstance(raw_response, dict):
            return {}

        usage = raw_response.get("usage", {})
        # NOTE: other model providers that use the OpenAI client may not report usage
        if usage is None:
            return {}

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> MultiModalCompletionResponse:
        message_dict = self._get_multi_modal_input_dict(prompt, image_documents)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=message_dict,
            max_tokens=self.max_new_tokens,
        )

        return MultiModalCompletionResponse(
            text=response.choices[0].message.content,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> MultiModalCompletionResponseGen:
        raise NotImplementedError

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> MultiModalCompletionResponse:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> MultiModalCompletionResponseAsyncGen:
        raise NotImplementedError
