"""Google's Gemini multi-modal models."""

import os
import typing
from typing import Any, Dict, Optional, Sequence

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.legacy.llms.gemini_utils import (
    ROLES_FROM_GEMINI,
    chat_from_gemini_response,
    chat_message_to_gemini,
    completion_from_gemini_response,
)
from llama_index.legacy.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.legacy.schema import ImageDocument

if typing.TYPE_CHECKING:
    import google.generativeai as genai

# PIL is imported lazily in the ctor but referenced throughout the module.
try:
    import PIL
except ImportError:
    # Swallow the error here, it's raised in the constructor where intent is clear.
    pass

# This lists the multi-modal models - see also llms.gemini for text models.
GEMINI_MM_MODELS = (
    "models/gemini-pro-vision",
    "models/gemini-ultra-vision",
)


class GeminiMultiModal(MultiModalLLM):
    """Gemini multimodal."""

    model_name: str = Field(
        default=GEMINI_MM_MODELS[0], description="The Gemini model to use."
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
        model_name: Optional[str] = GEMINI_MM_MODELS[0],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        generation_config: Optional["genai.types.GenerationConfigDict"] = None,
        safety_settings: "genai.types.SafetySettingOptions" = None,
        api_base: Optional[str] = None,
        transport: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
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
        try:
            import PIL  # noqa: F401
        except ImportError:
            raise ValueError(
                "Multi-modal support requires PIL. Please install it with "
                "`pip install pillow`."
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
        final_gen_config = {"temperature": temperature} | base_gen_config

        # Check whether the Gemini Model is supported or not
        if model_name not in GEMINI_MM_MODELS:
            raise ValueError(
                f"Invalid model {model_name}. "
                f"Available models are: {GEMINI_MM_MODELS}"
            )

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
        return "Gemini_MultiModal_LLM"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        total_tokens = self._model_meta.input_token_limit + self.max_tokens
        return MultiModalLLMMetadata(
            context_window=total_tokens,
            num_output=self.max_tokens,
            model_name=self.model_name,
        )

    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        images = [PIL.Image.open(doc.resolve_image()) for doc in image_documents]
        result = self._model.generate_content([prompt, *images], **kwargs)
        return completion_from_gemini_response(result)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseGen:
        images = [PIL.Image.open(doc.resolve_image()) for doc in image_documents]
        result = self._model.generate_content([prompt, *images], stream=True, **kwargs)
        yield from map(completion_from_gemini_response, result)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        *history, next_msg = map(chat_message_to_gemini, messages)
        chat = self._model.start_chat(history=history)
        response = chat.send_message(next_msg)
        return chat_from_gemini_response(response)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        *history, next_msg = map(chat_message_to_gemini, messages)
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

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        images = [PIL.Image.open(doc.resolve_image()) for doc in image_documents]
        result = await self._model.generate_content_async([prompt, *images], **kwargs)
        return completion_from_gemini_response(result)

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        images = [PIL.Image.open(doc.resolve_image()) for doc in image_documents]
        ait = await self._model.generate_content_async(
            [prompt, *images], stream=True, **kwargs
        )

        async def gen() -> CompletionResponseAsyncGen:
            async for comp in ait:
                yield completion_from_gemini_response(comp)

        return gen()

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        *history, next_msg = map(chat_message_to_gemini, messages)
        chat = self._model.start_chat(history=history)
        response = await chat.send_message_async(next_msg)
        return chat_from_gemini_response(response)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        *history, next_msg = map(chat_message_to_gemini, messages)
        chat = self._model.start_chat(history=history)
        response = await chat.send_message_async(next_msg, stream=True)

        async def gen() -> ChatResponseAsyncGen:
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
