from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
)

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.llms.openai_utils import from_openai_message, to_openai_message_dicts
from llama_index.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.multi_modal_llms.openai_utils import (
    generate_openai_multi_modal_chat_message,
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

    def _get_multi_modal_chat_messages(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[ImageDocument],
        **kwargs: Any,
    ) -> List[ChatCompletionMessageParam]:
        return to_openai_message_dicts(
            [
                generate_openai_multi_modal_chat_message(
                    prompt=prompt,
                    role=role,
                    image_documents=image_documents,
                    image_detail=self.image_detail,
                )
            ]
        )

    # Model Params for OpenAI GPT4V model.
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

    def _complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )
        response = self._client.chat.completions.create(
            messages=message_dict,
            stream=False,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.choices[0].message.content,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dicts = to_openai_message_dicts(messages)
        response = self._client.chat.completions.create(
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        openai_message = response.choices[0].message
        message = from_openai_message(openai_message)

        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        def gen() -> CompletionResponseGen:
            text = ""

            for response in self._client.chat.completions.create(
                messages=message_dict,
                stream=True,
                **all_kwargs,
            ):
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) > 0:
                    delta = response.choices[0].delta
                else:
                    delta = ChoiceDelta()

                # update using deltas
                content_delta = delta.content or ""
                text += content_delta

                yield CompletionResponse(
                    delta=content_delta,
                    text=text,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        message_dicts = to_openai_message_dicts(messages)

        def gen() -> ChatResponseGen:
            content = ""
            tool_calls: List[ChoiceDeltaToolCall] = []

            is_function = False
            for response in self._client.chat.completions.create(
                messages=message_dicts,
                stream=True,
                **self._get_model_kwargs(**kwargs),
            ):
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) > 0:
                    delta = response.choices[0].delta
                else:
                    delta = ChoiceDelta()

                # check if this chunk is the start of a function call
                if delta.tool_calls:
                    is_function = True

                # update using deltas
                role = delta.role or MessageRole.ASSISTANT
                content_delta = delta.content or ""
                content += content_delta

                additional_kwargs = {}
                if is_function:
                    tool_calls = self._update_tool_calls(tool_calls, delta.tool_calls)
                    additional_kwargs["tool_calls"] = tool_calls

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content_delta,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt, image_documents, **kwargs)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseGen:
        return self._stream_complete(prompt, image_documents, **kwargs)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return self._chat(messages, **kwargs)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        return self._stream_chat(messages, **kwargs)

    # ===== Async Endpoints =====

    async def _acomplete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )
        response = self._client.chat.completions.create(
            messages=message_dict,
            stream=False,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.choices[0].message.content,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        return await self._acomplete(prompt, image_documents, **kwargs)

    async def _astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        async def gen() -> CompletionResponseAsyncGen:
            text = ""

            for response in self._client.chat.completions.create(
                messages=message_dict,
                stream=True,
                **all_kwargs,
            ):
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) > 0:
                    delta = response.choices[0].delta
                else:
                    delta = ChoiceDelta()

                # update using deltas
                content_delta = delta.content or ""
                text += content_delta

                yield CompletionResponse(
                    delta=content_delta,
                    text=text,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dicts = to_openai_message_dicts(messages)
        response = self._client.chat.completions.create(
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        openai_message = response.choices[0].message
        message = from_openai_message(openai_message)

        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        message_dicts = to_openai_message_dicts(messages)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            tool_calls: List[ChoiceDeltaToolCall] = []

            is_function = False
            for response in self._client.chat.completions.create(
                messages=message_dicts,
                stream=True,
                **self._get_model_kwargs(**kwargs),
            ):
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) > 0:
                    delta = response.choices[0].delta
                else:
                    delta = ChoiceDelta()

                # check if this chunk is the start of a function call
                if delta.tool_calls:
                    is_function = True

                # update using deltas
                role = delta.role or MessageRole.ASSISTANT
                content_delta = delta.content or ""
                content += content_delta

                additional_kwargs = {}
                if is_function:
                    tool_calls = self._update_tool_calls(tool_calls, delta.tool_calls)
                    additional_kwargs["tool_calls"] = tool_calls

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content_delta,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await self._astream_complete(prompt, image_documents, **kwargs)

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return await self._achat(messages, **kwargs)

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        return await self._astream_chat(messages, **kwargs)
