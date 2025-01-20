"""Google's hosted Gemini API."""

import os
import warnings
from typing import Any, Dict, Optional, Sequence, cast

import google.generativeai as genai
from google.generativeai.types import generation_types
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.utilities.gemini_utils import (
    ROLES_FROM_GEMINI,
    merge_neighboring_same_role_messages,
)

from .utils import (
    chat_from_gemini_response,
    chat_message_to_gemini,
    completion_from_gemini_response,
)

GEMINI_MODELS = (
    "models/gemini-2.0-flash-exp",
    # Gemini 1.0 Pro Vision has been deprecated on July 12, 2024.
    # According to official recommendations, switch the default model to gemini-1.5-flash
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-latest",
    "models/gemini-pro",
    "models/gemini-pro-latest",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.0-pro",
    # for some reason, google lists this without the models prefix
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.0-pro",
)


class Gemini(CustomLLM):
    """
    Gemini LLM.

    Examples:
        `pip install llama-index-llms-gemini`

        ```python
        from llama_index.llms.gemini import Gemini

        llm = Gemini(model="models/gemini-ultra", api_key="YOUR_API_KEY")
        resp = llm.complete("Write a poem about a magic backpack")
        print(resp)
        ```
    """

    model: str = Field(default=GEMINI_MODELS[0], description="The Gemini model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    generate_kwargs: dict = Field(
        default_factory=dict, description="Kwargs for generation."
    )

    _model: genai.GenerativeModel = PrivateAttr()
    _model_meta: genai.types.Model = PrivateAttr()
    _request_options: Optional[genai.types.RequestOptions] = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODELS[0],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        generation_config: Optional[genai.types.GenerationConfigDict] = None,
        safety_settings: Optional[genai.types.SafetySettingDict] = None,
        callback_manager: Optional[CallbackManager] = None,
        api_base: Optional[str] = None,
        transport: Optional[str] = None,
        model_name: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        request_options: Optional[genai.types.RequestOptions] = None,
        **generate_kwargs: Any,
    ):
        """Creates a new Gemini model interface."""
        if model_name is not None:
            warnings.warn(
                "model_name is deprecated, please use model instead",
                DeprecationWarning,
            )

            model = model_name

        # API keys are optional. The API can be authorised via OAuth (detected
        # environmentally) or by the GOOGLE_API_KEY environment variable.
        config_params: Dict[str, Any] = {
            "api_key": api_key or os.getenv("GOOGLE_API_KEY"),
        }
        if api_base:
            config_params["client_options"] = {"api_endpoint": api_base}
        if transport:
            config_params["transport"] = transport
        if default_headers:
            default_metadata = []
            for key, value in default_headers.items():
                default_metadata.append((key, value))
            # `default_metadata` contains (key, value) pairs that will be sent with every request.
            # When using `transport="rest"`, these will be sent as HTTP headers.
            config_params["default_metadata"] = default_metadata
        # transport: A string, one of: [`rest`, `grpc`, `grpc_asyncio`].
        genai.configure(**config_params)

        base_gen_config = generation_config if generation_config else {}
        # Explicitly passed args take precedence over the generation_config.
        final_gen_config = cast(
            generation_types.GenerationConfigDict,
            {"temperature": temperature, **base_gen_config},
        )

        model_meta = genai.get_model(model)

        genai_model = genai.GenerativeModel(
            model_name=model,
            generation_config=final_gen_config,
            safety_settings=safety_settings,
        )

        supported_methods = model_meta.supported_generation_methods
        if "generateContent" not in supported_methods:
            raise ValueError(
                f"Model {model} does not support content generation, only "
                f"{supported_methods}."
            )

        if not max_tokens:
            max_tokens = model_meta.output_token_limit
        else:
            max_tokens = min(max_tokens, model_meta.output_token_limit)

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            generate_kwargs=generate_kwargs,
            callback_manager=callback_manager,
        )

        self._model_meta = model_meta
        self._model = genai_model
        self._request_options = request_options

    @classmethod
    def class_name(cls) -> str:
        return "Gemini_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        total_tokens = self._model_meta.input_token_limit + self.max_tokens
        return LLMMetadata(
            context_window=total_tokens,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        request_options = self._request_options or kwargs.pop("request_options", None)
        result = self._model.generate_content(
            prompt, request_options=request_options, **kwargs
        )
        return completion_from_gemini_response(result)

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        request_options = self._request_options or kwargs.pop("request_options", None)
        result = await self._model.generate_content_async(
            prompt, request_options=request_options, **kwargs
        )
        return completion_from_gemini_response(result)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        request_options = self._request_options or kwargs.pop("request_options", None)
        it = self._model.generate_content(
            prompt, stream=True, request_options=request_options, **kwargs
        )
        yield from map(completion_from_gemini_response, it)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        request_options = self._request_options or kwargs.pop("request_options", None)
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history)
        response = chat.send_message(
            next_msg,
            request_options=request_options,
            **kwargs,
        )
        return chat_from_gemini_response(response)

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        request_options = self._request_options or kwargs.pop("request_options", None)
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history)
        response = await chat.send_message_async(
            next_msg, request_options=request_options, **kwargs
        )
        return chat_from_gemini_response(response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        request_options = self._request_options or kwargs.pop("request_options", None)
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history)
        response = chat.send_message(
            next_msg, stream=True, request_options=request_options, **kwargs
        )

        def gen() -> ChatResponseGen:
            content = ""
            for r in response:
                top_candidate = r.candidates[0]
                content_delta = top_candidate.content.parts[0].text
                role = ROLES_FROM_GEMINI[top_candidate.content.role]
                raw = {
                    **(type(top_candidate).to_dict(top_candidate)),  # type: ignore
                    **(
                        type(response.prompt_feedback).to_dict(response.prompt_feedback)  # type: ignore
                    ),
                }
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=raw,
                )

        return gen()

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        request_options = self._request_options or kwargs.pop("request_options", None)
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, messages)
        chat = self._model.start_chat(history=history)
        response = await chat.send_message_async(
            next_msg, stream=True, request_options=request_options, **kwargs
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            async for r in response:
                top_candidate = r.candidates[0]
                content_delta = top_candidate.content.parts[0].text
                role = ROLES_FROM_GEMINI[top_candidate.content.role]
                raw = {
                    **(type(top_candidate).to_dict(top_candidate)),  # type: ignore
                    **(
                        type(response.prompt_feedback).to_dict(response.prompt_feedback)  # type: ignore
                    ),
                }
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=raw,
                )

        return gen()
