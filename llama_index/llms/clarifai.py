from typing import Any, Dict, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.llms.anthropic_utils import (
    anthropic_modelname_to_contextsize, messages_to_anthropic_prompt)
from llama_index.llms.base import (LLM, ChatMessage, ChatResponse,
                                   ChatResponseAsyncGen, ChatResponseGen,
                                   CompletionResponse,
                                   CompletionResponseAsyncGen,
                                   CompletionResponseGen, LLMMetadata,
                                   MessageRole, llm_chat_callback,
                                   llm_completion_callback)
from llama_index.llms.generic_utils import (
    achat_to_completion_decorator, astream_chat_to_completion_decorator,
    chat_to_completion_decorator, stream_chat_to_completion_decorator)

EXAMPLE_URL = "https://clarifai.com/anthropic/completion/models/claude-v2"


class Clarifai(LLM):
    model_url: Optional[str] = Field(
        description=f"Full URL of the model. e.g. `{EXAMPLE_URL}`"
    )
    model_id: Optional[str] = Field(description="Model ID.")
    model_version_id: Optional[str] = Field(description="Model Version ID.")
    app_id: Optional[str] = Field(description="Clarifai application ID of the model.")
    user_id: Optional[str] = Field(description="Clarifai user ID of the model.")

    _model: Any = PrivateAttr()
    _is_chat_model: bool = PrivateAttr()

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_url: Optional[str] = None,
        model_version_id: Optional[str] = "",
        app_id: Optional[str] = None,
        user_id: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        try:
            from clarifai.client.model import Model
            from google.protobuf.struct_pb2 import Struct
        except ImportError:
            raise ImportError("ClarifaiLLM requires `pip install clarifai`.")

        if model_url is not None and model_name is not None:
            raise ValueError("You can only specify one of model_url or model_name.")
        if model_url is None and model_name is None:
            raise ValueError("You must specify one of model_url or model_name.")

        if model_url is not None:
            self._model = Model(model_url)

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
                )

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
            model_name=self.model,
            is_chat_model=self._is_chat_model,
        )

    """
    Example:
    
    from clarifai.client.model import Model
    from google.protobuf.struct_pb2 import Struct


    inference_params = Struct()
    inference_params.update(dict(temperature=str(0.7)))

    output_info = dict(params=inference_params)

    m = Model(user_id="openai",app_id="chat-completion",model_id="GPT-4",
            model_version=dict(output_info=output_info))

    print(m.predict_by_bytes(b"Tweet on enjoying event at PyCon", "text"))
    """

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM."""
        prompt = magic(messages)  ## TODO: implement this converter message
        self._model.predict_by_bytes(prompt.encode(encoding="UTF-8"), "text")

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint for LLM."""
        pass

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError(
            "Clarifai does not currently support streaming completion."
        )

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError(
            "Clarifai does not currently support streaming completion."
        )
