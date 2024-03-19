import logging
from typing import Any, Optional, Sequence
import torch
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.types import PydanticProgramMode
from llama_index.llms.modelscope.utils import (
    chat_message_to_modelscope_messages,
    text_to_completion_response,
    modelscope_message_to_chat_response,
)
from modelscope import pipeline

DEFAULT_MODELSCOPE_MODEL = "qwen/Qwen-7B-Chat"
DEFAULT_MODELSCOPE_MODEL_REVISION = "master"
DEFAULT_MODELSCOPE_TASK = "chat"
DEFAULT_MODELSCOPE_DTYPE = "float16"
logger = logging.getLogger(__name__)

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class ModelScopeLLM(CustomLLM):
    """ModelScope LLM."""

    model_name: str = Field(
        default=DEFAULT_MODELSCOPE_MODEL,
        description=(
            "The model name to use from ModelScope. "
            "Unused if `model` is passed in directly."
        ),
    )
    model_revision: str = Field(
        default=DEFAULT_MODELSCOPE_MODEL_REVISION,
        description=(
            "The model revision to use from ModelScope. "
            "Unused if `model` is passed in directly."
        ),
    )
    task_name: str = Field(
        default=DEFAULT_MODELSCOPE_TASK,
        description=("The ModelScope task type, for llm use default chat."),
    )
    dtype: str = Field(
        default=DEFAULT_MODELSCOPE_DTYPE,
        description=("The ModelScope task type, for llm use default chat."),
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of tokens available for input.",
        gt=0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    system_prompt: str = Field(
        default="",
        description=(
            "The system prompt, containing any extra instructions or context. "
            "The model card on ModelScope should specify if this is needed."
        ),
    )
    query_wrapper_prompt: PromptTemplate = Field(
        default=PromptTemplate("{query_str}"),
        description=(
            "The query wrapper prompt, containing the query placeholder. "
            "The model card on ModelScope should specify if this is needed. "
            "Should contain a `{query_str}` placeholder."
        ),
    )
    device_map: str = Field(
        default="auto", description="The device_map to use. Defaults to 'auto'."
    )
    tokenizer_kwargs: dict = Field(
        default_factory=dict, description="The kwargs to pass to the tokenizer."
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during initialization.",
    )
    generate_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during generation.",
    )

    _pipeline: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_MODELSCOPE_MODEL,
        model_revision: str = DEFAULT_MODELSCOPE_MODEL_REVISION,
        task_name: str = DEFAULT_MODELSCOPE_TASK,
        dtype: str = DEFAULT_MODELSCOPE_DTYPE,
        model: Optional[Any] = None,
        device_map: Optional[str] = "auto",
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        callback_manager: Optional[CallbackManager] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    ) -> None:
        """Initialize params."""
        model_kwargs = model_kwargs or {}
        if model:
            self._pipeline = model
        else:
            self._pipeline = pipeline(
                task=task_name,
                model=model_name,
                model_revision=model_revision,
                llm_first=True,
                torch_dtype=_STR_DTYPE_TO_TORCH_DTYPE[dtype],
                device_map=device_map,
            )

        super().__init__(
            model_kwargs=model_kwargs or {},
            generate_kwargs=generate_kwargs or {},
            callback_manager=callback_manager,
            pydantic_program_mode=pydantic_program_mode,
        )

    @classmethod
    def class_name(cls) -> str:
        return "ModelScope_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=None,
            num_output=None,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return text_to_completion_response(self._pipeline(prompt, **kwargs))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        yield self.complete(prompt, **kwargs)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return modelscope_message_to_chat_response(
            self._pipeline(chat_message_to_modelscope_messages(messages), **kwargs)
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        yield self.chat(messages, **kwargs)
