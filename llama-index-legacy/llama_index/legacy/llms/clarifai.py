from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.legacy.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.legacy.llms.llm import LLM
from llama_index.legacy.types import BaseOutputParser, PydanticProgramMode

EXAMPLE_URL = "https://clarifai.com/anthropic/completion/models/claude-v2"


class Clarifai(LLM):
    model_url: Optional[str] = Field(
        description=f"Full URL of the model. e.g. `{EXAMPLE_URL}`"
    )
    model_version_id: Optional[str] = Field(description="Model Version ID.")
    app_id: Optional[str] = Field(description="Clarifai application ID of the model.")
    user_id: Optional[str] = Field(description="Clarifai user ID of the model.")
    pat: Optional[str] = Field(
        description="Personal Access Tokens(PAT) to validate requests."
    )

    _model: Any = PrivateAttr()
    _is_chat_model: bool = PrivateAttr()

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_url: Optional[str] = None,
        model_version_id: Optional[str] = "",
        app_id: Optional[str] = None,
        user_id: Optional[str] = None,
        pat: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ):
        try:
            import os

            from clarifai.client.model import Model
        except ImportError:
            raise ImportError("ClarifaiLLM requires `pip install clarifai`.")

        if pat is None and os.environ.get("CLARIFAI_PAT") is not None:
            pat = os.environ.get("CLARIFAI_PAT")

        if not pat and os.environ.get("CLARIFAI_PAT") is None:
            raise ValueError(
                "Set `CLARIFAI_PAT` as env variable or pass `pat` as constructor argument"
            )

        if model_url is not None and model_name is not None:
            raise ValueError("You can only specify one of model_url or model_name.")
        if model_url is None and model_name is None:
            raise ValueError("You must specify one of model_url or model_name.")

        if model_name is not None:
            if app_id is None or user_id is None:
                raise ValueError(
                    f"Missing one app ID or user ID of the model: {app_id=}, {user_id=}"
                )
            else:
                self._model = Model(
                    user_id=user_id,
                    app_id=app_id,
                    model_id=model_name,
                    model_version={"id": model_version_id},
                    pat=pat,
                )

        if model_url is not None:
            self._model = Model(model_url, pat=pat)
            model_name = self._model.id

        self._is_chat_model = False
        if "chat" in self._model.app_id or "chat" in self._model.id:
            self._is_chat_model = True

        additional_kwargs = additional_kwargs or {}

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            model_name=model_name,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "ClarifaiLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self._model,
            is_chat_model=self._is_chat_model,
        )

    # TODO: When the Clarifai python SDK supports inference params, add here.
    def chat(
        self,
        messages: Sequence[ChatMessage],
        inference_params: Optional[Dict] = {},
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat endpoint for LLM."""
        prompt = "".join([str(m) for m in messages])
        try:
            response = (
                self._model.predict_by_bytes(
                    input_bytes=prompt.encode(encoding="UTF-8"),
                    input_type="text",
                    inference_params=inference_params,
                )
                .outputs[0]
                .data.text.raw
            )
        except Exception as e:
            raise Exception(f"Prediction failed: {e}")
        return ChatResponse(message=ChatMessage(content=response))

    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        inference_params: Optional[Dict] = {},
        **kwargs: Any,
    ) -> CompletionResponse:
        """Completion endpoint for LLM."""
        try:
            response = (
                self._model.predict_by_bytes(
                    input_bytes=prompt.encode(encoding="utf-8"),
                    input_type="text",
                    inference_params=inference_params,
                )
                .outputs[0]
                .data.text.raw
            )
        except Exception as e:
            raise Exception(f"Prediction failed: {e}")
        return CompletionResponse(text=response)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError(
            "Clarifai does not currently support streaming completion."
        )

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError(
            "Clarifai does not currently support streaming completion."
        )

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError("Currently not supported.")

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError("Currently not supported.")

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError("Clarifai does not currently support this function.")
