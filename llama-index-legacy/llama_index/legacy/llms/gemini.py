"""Google's hosted Gemini API."""

import os
import typing
from typing import Any, Dict, Optional, Sequence

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.legacy.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.legacy.llms.custom import CustomLLM
from llama_index.legacy.llms.gemini_utils import (
    ROLES_FROM_GEMINI,
    chat_from_gemini_response,
    chat_message_to_gemini,
    completion_from_gemini_response,
    merge_neighboring_same_role_messages,
)

if typing.TYPE_CHECKING:
    import google.generativeai as genai


GEMINI_MODELS = (
    "models/gemini-pro",
    "models/gemini-ultra",
)


class Gemini(CustomLLM):
    """Gemini."""

    model_name: str = Field(
        default=GEMINI_MODELS[0], description="The Gemini model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    generate_kwargs: dict = Field(
        default_factory=dict, description="Kwargs for generation."
    )

    _model: "genai.GenerativeModel" = PrivateAttr()
    _model_meta: "genai.types.Model" = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = GEMINI_MODELS[0],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        generation_config: Optional["genai.types.GenerationConfigDict"] = None,
        safety_settings: "genai.types.SafetySettingOptions" = None,
        callback_manager: Optional[CallbackManager] = None,
        api_base: Optional[str] = None,
        transport: Optional[str] = None,
        **generate_kwargs: Any,
    ):
        """Creates a new Gemini model interface."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ValueError(
                "Gemini is not installed. Please install it with "
                "`pip install 'google-generativeai>=0.3.0'`."
            )

        # API keys are optional. The API can be authorised via OAuth (detected
        # environmentally) or by the GOOGLE_API_KEY environment variable.
        config_params: Dict[str, Any] = {
            "api_key": api_key or os.getenv("GOOGLE_API_KEY"),
        }
        if api_base:
            config_params["client_options"] = {"api_endpoint": api_base}
        if transport:
            config_params["transport"] = transport
        # transport: A string, one of: [`rest`, `grpc`, `grpc_asyncio`].
        genai.configure(**config_params)

        base_gen_config = generation_config if generation_config else {}
        # Explicitly passed args take precedence over the generation_config.
        final_gen_config = {"temperature": temperature, **base_gen_config}

        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=final_gen_config,
            safety_settings=safety_settings,
        )

        self._model_meta = genai.get_model(model_name)

        supported_methods = self._model_meta.supported_generation_methods
        if "generateContent" not in supported_methods:
            raise ValueError(
                f"Model {model_name} does not support content generation, only "
                f"{supported_methods}."
            )

        if not max_tokens:
            max_tokens = self._model_meta.output_token_limit
        else:
            max_tokens = min(max_tokens, self._model_meta.output_token_limit)

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            generate_kwargs=generate_kwargs,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Gemini_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        total_tokens = self._model_meta.input_token_limit + self.max_tokens
        return LLMMetadata(
            context_window=total_tokens,
            num_output=self.max_tokens,
            model_name=self.model_name,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        result = self._model.generate_content(prompt, **kwargs)
        return completion_from_gemini_response(result)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        it = self._model.generate_content(prompt, stream=True, **kwargs)
        yield from map(completion_from_gemini_response, it)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history)
        response = chat.send_message(next_msg)
        return chat_from_gemini_response(response)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history)
        response = chat.send_message(next_msg, stream=True)

        def gen() -> ChatResponseGen:
            content = ""
            for r in response:
                top_candidate = r.candidates[0]
                content_delta = top_candidate.content.parts[0].text
                role = ROLES_FROM_GEMINI[top_candidate.content.role]
                raw = {
                    **(type(top_candidate).to_dict(top_candidate)),
                    **(
                        type(response.prompt_feedback).to_dict(response.prompt_feedback)
                    ),
                }
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=raw,
                )

        return gen()
