from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import tiktoken

from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager
from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.generic_utils import (
    achat_to_completion_decorator,
    acompletion_to_chat_decorator,
    astream_chat_to_completion_decorator,
    astream_completion_to_chat_decorator,
    chat_to_completion_decorator,
    completion_to_chat_decorator,
    stream_chat_to_completion_decorator,
    stream_completion_to_chat_decorator,
)
from llama_index.llms.openai_utils import (
    acompletion_with_retry,
    completion_with_retry,
    from_openai_message_dict,
    is_chat_model,
    is_function_calling_model,
    openai_modelname_to_contextsize,
    resolve_openai_credentials,
    to_openai_message_dicts,
)


@runtime_checkable
class Tokenizer(Protocol):
    """Tokenizers support an encode function that returns a list of ints."""

    def encode(self, text: str) -> List[int]:
        ...


class OpenAI(LLM):
    model: str = Field(description="The OpenAI model to use.")
    temperature: float = Field(description="The temperature to use during generation.")
    max_tokens: Optional[int] = Field(
        default=None, description="The maximum number of tokens to generate."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )
    max_retries: int = Field(description="The maximum number of API retries.")

    api_key: str = Field(default=None, description="The OpenAI API key.", exclude=True)
    api_type: str = Field(default=None, description="The OpenAI API type.")
    api_base: str = Field(description="The base URL for OpenAI API.")
    api_version: str = Field(description="The API version for OpenAI API.")

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        api_type: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        api_key, api_type, api_base, api_version = resolve_openai_credentials(
            api_key=api_key,
            api_type=api_type,
            api_base=api_base,
            api_version=api_version,
        )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
            api_key=api_key,
            api_type=api_type,
            api_version=api_version,
            api_base=api_base,
            **kwargs,
        )

    def _get_model_name(self) -> str:
        model_name = self.model
        if "ft-" in model_name:  # legacy fine-tuning
            model_name = model_name.split(":")[0]
        elif model_name.startswith("ft:"):
            model_name = model_name.split(":")[1]
        return model_name

    @classmethod
    def class_name(cls) -> str:
        return "openai_llm"

    @property
    def _tokenizer(self) -> Tokenizer:
        return tiktoken.encoding_for_model(self._get_model_name())

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=openai_modelname_to_contextsize(self._get_model_name()),
            num_output=self.max_tokens or -1,
            is_chat_model=is_chat_model(model=self._get_model_name()),
            is_function_calling_model=is_function_calling_model(
                model=self._get_model_name()
            ),
            model_name=self.model,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self._use_chat_completions(kwargs):
            chat_fn = self._chat
        else:
            chat_fn = completion_to_chat_decorator(self._complete)
        return chat_fn(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if self._use_chat_completions(kwargs):
            stream_chat_fn = self._stream_chat
        else:
            stream_chat_fn = stream_completion_to_chat_decorator(self._stream_complete)
        return stream_chat_fn(messages, **kwargs)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if self._use_chat_completions(kwargs):
            complete_fn = chat_to_completion_decorator(self._chat)
        else:
            complete_fn = self._complete
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        if self._use_chat_completions(kwargs):
            stream_complete_fn = stream_chat_to_completion_decorator(self._stream_chat)
        else:
            stream_complete_fn = self._stream_complete
        return stream_complete_fn(prompt, **kwargs)

    def _use_chat_completions(self, kwargs: Dict[str, Any]) -> bool:
        if "use_chat_completions" in kwargs:
            return kwargs["use_chat_completions"]
        return self.metadata.is_chat_model

    @property
    def _credential_kwargs(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "api_type": self.api_type,
            "api_base": self.api_base,
            "api_version": self.api_version,
        }

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            # If max_tokens is None, don't include in the payload:
            # https://platform.openai.com/docs/api-reference/chat
            # https://platform.openai.com/docs/api-reference/completions
            base_kwargs["max_tokens"] = self.max_tokens
        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Get all data for the request as a dictionary."""
        return {**self._credential_kwargs, **self._model_kwargs, **kwargs}

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        message_dicts = to_openai_message_dicts(messages)
        response = completion_with_retry(
            is_chat_model=True,
            max_retries=self.max_retries,
            messages=message_dicts,
            stream=False,
            **self._get_all_kwargs(**kwargs),
        )
        message_dict = response["choices"][0]["message"]
        message = from_openai_message_dict(message_dict)

        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        message_dicts = to_openai_message_dicts(messages)

        def gen() -> ChatResponseGen:
            content = ""
            function_call: Optional[dict] = None
            for response in completion_with_retry(
                is_chat_model=True,
                max_retries=self.max_retries,
                messages=message_dicts,
                stream=True,
                **self._get_all_kwargs(**kwargs),
            ):
                if len(response["choices"]) == 0 and (
                    response.get("prompt_annotations")
                    or response.get("prompt_filter_results")
                ):
                    # When asking a stream response from the Azure OpenAI API
                    # you first get an empty message with the content filtering
                    # results. Ignore this message
                    continue

                if len(response["choices"]) > 0:
                    delta = response["choices"][0]["delta"]
                else:
                    delta = {}
                role = delta.get("role", "assistant")
                content_delta = delta.get("content", "") or ""
                content += content_delta

                function_call_delta = delta.get("function_call", None)
                if function_call_delta is not None:
                    if function_call is None:
                        function_call = function_call_delta

                        ## ensure we do not add a blank function call
                        if function_call.get("function_name", "") is None:
                            del function_call["function_name"]
                    else:
                        function_call["arguments"] = (
                            function_call.get("arguments", "")
                            + function_call_delta["arguments"]
                        )

                additional_kwargs = {}
                if function_call is not None:
                    additional_kwargs["function_call"] = function_call

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

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        self._update_max_tokens(all_kwargs, prompt)

        response = completion_with_retry(
            is_chat_model=False,
            max_retries=self.max_retries,
            prompt=prompt,
            stream=False,
            **all_kwargs,
        )
        text = response["choices"][0]["text"]
        return CompletionResponse(
            text=text,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        self._update_max_tokens(all_kwargs, prompt)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in completion_with_retry(
                is_chat_model=False,
                max_retries=self.max_retries,
                prompt=prompt,
                stream=True,
                **all_kwargs,
            ):
                if len(response["choices"]) > 0:
                    delta = response["choices"][0]["text"]
                else:
                    delta = ""
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    def _update_max_tokens(self, all_kwargs: Dict[str, Any], prompt: str) -> None:
        if self.max_tokens is not None:
            return
        # NOTE: non-chat completion endpoint requires max_tokens to be set
        context_window = self.metadata.context_window
        tokens = self._tokenizer.encode(prompt)
        max_tokens = context_window - len(tokens)
        if max_tokens <= 0:
            raise ValueError(
                f"The prompt is too long for the model. "
                f"Please use a prompt that is less than {context_window} tokens."
            )
        all_kwargs["max_tokens"] = max_tokens

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

    # ===== Async Endpoints =====
    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        achat_fn: Callable[..., Awaitable[ChatResponse]]
        if self._use_chat_completions(kwargs):
            achat_fn = self._achat
        else:
            achat_fn = acompletion_to_chat_decorator(self._acomplete)
        return await achat_fn(messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        astream_chat_fn: Callable[..., Awaitable[ChatResponseAsyncGen]]
        if self._use_chat_completions(kwargs):
            astream_chat_fn = self._astream_chat
        else:
            astream_chat_fn = astream_completion_to_chat_decorator(
                self._astream_complete
            )
        return await astream_chat_fn(messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if self._use_chat_completions(kwargs):
            acomplete_fn = achat_to_completion_decorator(self._achat)
        else:
            acomplete_fn = self._acomplete
        return await acomplete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        if self._use_chat_completions(kwargs):
            astream_complete_fn = astream_chat_to_completion_decorator(
                self._astream_chat
            )
        else:
            astream_complete_fn = self._astream_complete
        return await astream_complete_fn(prompt, **kwargs)

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        message_dicts = to_openai_message_dicts(messages)
        response = await acompletion_with_retry(
            is_chat_model=True,
            max_retries=self.max_retries,
            messages=message_dicts,
            stream=False,
            **self._get_all_kwargs(**kwargs),
        )
        message_dict = response["choices"][0]["message"]
        message = from_openai_message_dict(message_dict)

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
            function_call: Optional[dict] = None
            async for response in await acompletion_with_retry(
                is_chat_model=True,
                max_retries=self.max_retries,
                messages=message_dicts,
                stream=True,
                **self._get_all_kwargs(**kwargs),
            ):
                if len(response["choices"]) == 0 and response.get("prompt_annotations"):
                    # open ai sends empty response first while streaming ignore it
                    continue
                if len(response["choices"]) > 0:
                    delta = response["choices"][0]["delta"]
                else:
                    delta = {}
                role = delta.get("role", "assistant")
                content_delta = delta.get("content", "") or ""
                content += content_delta

                function_call_delta = delta.get("function_call", None)
                if function_call_delta is not None:
                    if function_call is None:
                        function_call = function_call_delta

                        ## ensure we do not add a blank function call
                        if function_call.get("function_name", "") is None:
                            del function_call["function_name"]
                    else:
                        function_call["arguments"] = (
                            function_call.get("arguments", "")
                            + function_call_delta["arguments"]
                        )

                additional_kwargs = {}
                if function_call is not None:
                    additional_kwargs["function_call"] = function_call

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

    async def _acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        self._update_max_tokens(all_kwargs, prompt)

        response = await acompletion_with_retry(
            is_chat_model=False,
            max_retries=self.max_retries,
            prompt=prompt,
            stream=False,
            **all_kwargs,
        )
        text = response["choices"][0]["text"]
        return CompletionResponse(
            text=text,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    async def _astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        self._update_max_tokens(all_kwargs, prompt)

        async def gen() -> CompletionResponseAsyncGen:
            text = ""
            async for response in await acompletion_with_retry(
                is_chat_model=False,
                max_retries=self.max_retries,
                prompt=prompt,
                stream=True,
                **all_kwargs,
            ):
                if len(response["choices"]) > 0:
                    delta = response["choices"][0]["text"]
                else:
                    delta = ""
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()
