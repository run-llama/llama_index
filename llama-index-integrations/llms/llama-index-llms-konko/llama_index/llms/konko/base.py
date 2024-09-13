from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence

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
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    acompletion_to_chat_decorator,
    astream_chat_to_completion_decorator,
    astream_completion_to_chat_decorator,
    chat_to_completion_decorator,
    completion_to_chat_decorator,
    stream_chat_to_completion_decorator,
    stream_completion_to_chat_decorator,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.konko.utils import (
    acompletion_with_retry,
    completion_with_retry,
    from_openai_message_dict,
    is_openai_v1,
    resolve_konko_credentials,
    to_openai_message_dicts,
)

import konko

DEFAULT_KONKO_MODEL = "meta-llama/llama-2-13b-chat"


@dataclass
class ModelInfo:
    name: str
    max_context_length: int
    is_chat_model: bool


class Konko(LLM):
    """Konko LLM.

    Examples:
        `pip install llama-index-llms-konko`

        ```python
        import os
        from llama_index.llms.konko import Konko
        from llama_index.core.llms import ChatMessage

        # Set up the Konko LLM with the desired model
        llm = Konko(model="meta-llama/llama-2-13b-chat")

        # Set the Konko API key
        os.environ["KONKO_API_KEY"] = "<your-api-key>"

        # Create a ChatMessage object
        message = ChatMessage(role="user", content="Explain Big Bang Theory briefly")

        # Call the chat method with the ChatMessage object
        response = llm.chat([message])

        # Print the response
        print(response)
        ```
    """

    model: str = Field(
        default=DEFAULT_KONKO_MODEL, description="The konko model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: Optional[int] = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the konko API."
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", ge=0
    )

    konko_api_key: str = Field(default=None, description="The konko API key.")
    openai_api_key: str = Field(default=None, description="The Openai API key.")
    api_type: str = Field(default=None, description="The konko API type.")
    model_info_dict: Dict[str, ModelInfo]

    def __init__(
        self,
        model: str = DEFAULT_KONKO_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        konko_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        api_type: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        model_info_dict: Optional[Dict[str, ModelInfo]] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        (
            konko_api_key,
            openai_api_key,
            api_type,
            api_base,
            api_version,
        ) = resolve_konko_credentials(
            konko_api_key=konko_api_key,
            openai_api_key=openai_api_key,
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
            konko_api_key=konko_api_key,
            openai_api_key=openai_api_key,
            api_type=api_type,
            api_version=api_version,
            api_base=api_base,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            model_info_dict=self._create_model_info_dict(),
            **kwargs,
        )

    def _get_model_name(self) -> str:
        return self.model

    @classmethod
    def class_name(cls) -> str:
        return "Konko_LLM"

    def _create_model_info_dict(self) -> Dict[str, ModelInfo]:
        models_info_dict = {}
        if is_openai_v1():
            models = konko.models.list().data
            for model in models:
                model_info = ModelInfo(
                    name=model.name,
                    max_context_length=model.max_context_length,
                    is_chat_model=model.is_chat,
                )
                models_info_dict[model.name] = model_info
        else:
            models = konko.Model.list().data
            for model in models:
                model_info = ModelInfo(
                    name=model["name"],
                    max_context_length=model["max_context_length"],
                    is_chat_model=model["is_chat"],
                )
                models_info_dict[model["name"]] = model_info

        return models_info_dict

    def _get_model_info(self) -> ModelInfo:
        model_name = self._get_model_name()
        model_info = self.model_info_dict.get(model_name)
        if model_info is None:
            raise ValueError(
                f"Unknown model: {model_name}. Please provide a valid Konko model name. "
                "Known models are: " + ", ".join(self.model_info_dict.keys())
            )
        return model_info

    def _is_chat_model(self) -> bool:
        """
        Check if the specified model is a chat model.

        Args:
        - model_id (str): The ID of the model to check.

        Returns:
        - bool: True if the model is a chat model, False otherwise.

        Raises:
        - ValueError: If the model_id is not found in the list of models.
        """
        model_info = self._get_model_info()
        return model_info.is_chat_model

    @property
    def metadata(self) -> LLMMetadata:
        model_info = self._get_model_info()
        return LLMMetadata(
            context_window=model_info.max_context_length,
            num_output=self.max_tokens,
            is_chat_model=model_info.is_chat_model,
            model_name=self.model,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self._is_chat_model():
            chat_fn = self._chat
        else:
            chat_fn = completion_to_chat_decorator(self._complete)
        return chat_fn(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if self._is_chat_model():
            stream_chat_fn = self._stream_chat
        else:
            stream_chat_fn = stream_completion_to_chat_decorator(self._stream_complete)
        return stream_chat_fn(messages, **kwargs)

    @property
    def _credential_kwargs(self) -> Dict[str, Any]:
        return {
            "konko_api_key": self.konko_api_key,
            "api_type": self.api_type,
            "openai_api_key": self.openai_api_key,
        }

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if not self._is_chat_model():
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = completion_with_retry(
            is_chat_model=self._is_chat_model(),
            max_retries=self.max_retries,
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        if is_openai_v1():
            message_dict = response.choices[0].message
        else:
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
        if not self._is_chat_model():
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        def gen() -> ChatResponseGen:
            content = ""
            for response in completion_with_retry(
                is_chat_model=self._is_chat_model(),
                max_retries=self.max_retries,
                messages=message_dicts,
                stream=True,
                **all_kwargs,
            ):
                if is_openai_v1():
                    if len(response.choices) == 0 and response.prompt_annotations:
                        continue
                    delta = (
                        response.choices[0].delta if len(response.choices) > 0 else {}
                    )
                    role_value = delta.role
                    content_delta = delta.content or ""
                else:
                    if "choices" not in response or len(response["choices"]) == 0:
                        continue
                    delta = response["choices"][0].get("delta", {})
                    role_value = delta["role"]
                    content_delta = delta["content"] or ""

                role = role_value if role_value is not None else "assistant"
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                    ),
                    delta=content_delta,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        if self._is_chat_model():
            complete_fn = chat_to_completion_decorator(self._chat)
        else:
            complete_fn = self._complete
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        if self._is_chat_model():
            stream_complete_fn = stream_chat_to_completion_decorator(self._stream_chat)
        else:
            stream_complete_fn = self._stream_complete
        return stream_complete_fn(prompt, **kwargs)

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

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if self._is_chat_model():
            raise ValueError("This model is a chat model.")

        all_kwargs = self._get_all_kwargs(**kwargs)
        if self.max_tokens is None:
            # NOTE: non-chat completion endpoint requires max_tokens to be set
            max_tokens = self._get_max_token_for_prompt(prompt)
            all_kwargs["max_tokens"] = max_tokens

        response = completion_with_retry(
            is_chat_model=self._is_chat_model(),
            max_retries=self.max_retries,
            prompt=prompt,
            stream=False,
            **all_kwargs,
        )
        if is_openai_v1():
            text = response.choices[0].text
        else:
            text = response["choices"][0]["text"]

        return CompletionResponse(
            text=text,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        if self._is_chat_model():
            raise ValueError("This model is a chat model.")

        all_kwargs = self._get_all_kwargs(**kwargs)
        if self.max_tokens is None:
            # NOTE: non-chat completion endpoint requires max_tokens to be set
            max_tokens = self._get_max_token_for_prompt(prompt)
            all_kwargs["max_tokens"] = max_tokens

        def gen() -> CompletionResponseGen:
            text = ""
            for response in completion_with_retry(
                is_chat_model=self._is_chat_model(),
                max_retries=self.max_retries,
                prompt=prompt,
                stream=True,
                **all_kwargs,
            ):
                if is_openai_v1():
                    if len(response.choices) > 0:
                        delta = response.choices[0].text
                    else:
                        delta = ""
                else:
                    if len(response["choices"]) > 0:
                        delta = response["choices"][0].text
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

    def _get_max_token_for_prompt(self, prompt: str) -> int:
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Please install tiktoken to use the max_tokens=None feature."
            )
        context_window = self.metadata.context_window
        encoding = tiktoken.encoding_for_model(self._get_model_name())
        tokens = encoding.encode(prompt)
        max_token = context_window - len(tokens)
        if max_token <= 0:
            raise ValueError(
                f"The prompt is too long for the model. "
                f"Please use a prompt that is less than {context_window} tokens."
            )
        return max_token

    # ===== Async Endpoints =====
    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        achat_fn: Callable[..., Awaitable[ChatResponse]]
        if self._is_chat_model():
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
        if self._is_chat_model():
            astream_chat_fn = self._astream_chat
        else:
            astream_chat_fn = astream_completion_to_chat_decorator(
                self._astream_complete
            )
        return await astream_chat_fn(messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        if self._is_chat_model():
            acomplete_fn = achat_to_completion_decorator(self._achat)
        else:
            acomplete_fn = self._acomplete
        return await acomplete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        if self._is_chat_model():
            astream_complete_fn = astream_chat_to_completion_decorator(
                self._astream_chat
            )
        else:
            astream_complete_fn = self._astream_complete
        return await astream_complete_fn(prompt, **kwargs)

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        if not self._is_chat_model():
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = await acompletion_with_retry(
            is_chat_model=self._is_chat_model(),
            max_retries=self.max_retries,
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        if is_openai_v1:  # type: ignore
            message_dict = response.choices[0].message
        else:
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
        if not self._is_chat_model():
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            _function_call: Optional[dict] = None
            async for response in await acompletion_with_retry(
                is_chat_model=self._is_chat_model(),
                max_retries=self.max_retries,
                messages=message_dicts,
                stream=True,
                **all_kwargs,
            ):
                if is_openai_v1():
                    if len(response.choices) > 0:
                        delta = response.choices[0].delta
                    else:
                        delta = {}
                    role = delta.role
                    content_delta = delta.content
                else:
                    if len(response["choices"]) > 0:
                        delta = response["choices"][0].delta
                    else:
                        delta = {}
                    role = delta["role"]
                    content_delta = delta["content"]
                content += content_delta

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                    ),
                    delta=content_delta,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    async def _acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if self._is_chat_model():
            raise ValueError("This model is a chat model.")

        all_kwargs = self._get_all_kwargs(**kwargs)
        if self.max_tokens is None:
            # NOTE: non-chat completion endpoint requires max_tokens to be set
            max_tokens = self._get_max_token_for_prompt(prompt)
            all_kwargs["max_tokens"] = max_tokens

        response = await acompletion_with_retry(
            is_chat_model=self._is_chat_model(),
            max_retries=self.max_retries,
            prompt=prompt,
            stream=False,
            **all_kwargs,
        )
        if is_openai_v1():
            text = response.choices[0].text
        else:
            text = response["choices"][0]["text"]
        return CompletionResponse(
            text=text,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    async def _astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        if self._is_chat_model():
            raise ValueError("This model is a chat model.")

        all_kwargs = self._get_all_kwargs(**kwargs)
        if self.max_tokens is None:
            # NOTE: non-chat completion endpoint requires max_tokens to be set
            max_tokens = self._get_max_token_for_prompt(prompt)
            all_kwargs["max_tokens"] = max_tokens

        async def gen() -> CompletionResponseAsyncGen:
            text = ""
            async for response in await acompletion_with_retry(
                is_chat_model=self._is_chat_model(),
                max_retries=self.max_retries,
                prompt=prompt,
                stream=True,
                **all_kwargs,
            ):
                if is_openai_v1():
                    if len(response.choices) > 0:
                        delta = response.choices[0].text
                    else:
                        delta = ""
                else:
                    if len(response["choices"]) > 0:
                        delta = response["choices"][0].text
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
