from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, get_args

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.core.base.query_pipeline.query import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ConfigDict,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_INPUT_FILES,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.schema import BaseComponent, ImageDocument


class MultiModalLLMMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    context_window: Optional[int] = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "Total number of tokens the model can be input when generating a response."
        ),
    )
    num_output: Optional[int] = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of tokens the model can output when generating a response.",
    )
    num_input_files: Optional[int] = Field(
        default=DEFAULT_NUM_INPUT_FILES,
        description="Number of input files the model can take when generating a response.",
    )
    is_function_calling_model: Optional[bool] = Field(
        default=False,
        # SEE: https://openai.com/blog/function-calling-and-other-api-updates
        description=(
            "Set True if the model supports function calling messages, similar to"
            " OpenAI's function calling API. For example, converting 'Email Anya to"
            " see if she wants to get coffee next Friday' to a function call like"
            " `send_email(to: string, body: string)`."
        ),
    )
    model_name: str = Field(
        default="unknown",
        description=(
            "The model's name used for logging, testing, and sanity checking. For some"
            " models this can be automatically discerned. For other models, like"
            " locally loaded models, this must be manually specified."
        ),
    )

    is_chat_model: bool = Field(
        default=False,
        description=(
            "Set True if the model exposes a chat interface (i.e. can be passed a"
            " sequence of messages, rather than text), like OpenAI's"
            " /v1/chat/completions endpoint."
        ),
    )


class MultiModalLLM(ChainableMixin, BaseComponent, DispatcherSpanMixin):
    """Multi-Modal LLM interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    @property
    @abstractmethod
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi-Modal LLM metadata."""

    @abstractmethod
    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Stream chat endpoint for Multi-Modal LLM."""

    # ===== Async Endpoints =====

    @abstractmethod
    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint for Multi-Modal LLM."""

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """Return query component."""
        if self.metadata.is_chat_model:
            # TODO: we don't have a separate chat component
            return MultiModalCompleteComponent(multi_modal_llm=self, **kwargs)
        else:
            return MultiModalCompleteComponent(multi_modal_llm=self, **kwargs)

    def __init_subclass__(cls, **kwargs) -> None:
        """
        The callback decorators installs events, so they must be applied before
        the span decorators, otherwise the spans wouldn't contain the events.
        """
        for attr in (
            "complete",
            "acomplete",
            "stream_complete",
            "astream_complete",
            "chat",
            "achat",
            "stream_chat",
            "astream_chat",
        ):
            if callable(method := cls.__dict__.get(attr)):
                if attr.endswith("chat"):
                    setattr(cls, attr, llm_chat_callback()(method))
                else:
                    setattr(cls, attr, llm_completion_callback()(method))
        super().__init_subclass__(**kwargs)


class BaseMultiModalComponent(QueryComponent):
    """Base LLM component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    multi_modal_llm: MultiModalLLM = Field(..., description="LLM")
    streaming: bool = Field(default=False, description="Streaming mode")

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""
        # TODO: make callbacks work with multi-modal


class MultiModalCompleteComponent(BaseMultiModalComponent):
    """Multi-modal completion component."""

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "prompt" not in input:
            raise ValueError("Prompt must be in input dict.")

        # do special check to see if prompt is a list of chat messages
        if isinstance(input["prompt"], get_args(List[ChatMessage])):
            raise NotImplementedError(
                "Chat messages not yet supported as input to multi-modal model."
            )
        else:
            input["prompt"] = validate_and_convert_stringable(input["prompt"])

        # make sure image documents are valid
        if "image_documents" in input:
            if not isinstance(input["image_documents"], list):
                raise ValueError("image_documents must be a list.")
            for doc in input["image_documents"]:
                if not isinstance(doc, ImageDocument):
                    raise ValueError(
                        "image_documents must be a list of ImageDocument objects."
                    )

        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # TODO: support only complete for now
        prompt = kwargs["prompt"]
        image_documents = kwargs.get("image_documents", [])
        if self.streaming:
            response = self.multi_modal_llm.stream_complete(prompt, image_documents)
        else:
            response = self.multi_modal_llm.complete(prompt, image_documents)
        return {"output": response}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # TODO: support only complete for now
        # non-trivial to figure how to support chat/complete/etc.
        prompt = kwargs["prompt"]
        image_documents = kwargs.get("image_documents", [])
        if self.streaming:
            response = await self.multi_modal_llm.astream_complete(
                prompt, image_documents
            )
        else:
            response = await self.multi_modal_llm.acomplete(prompt, image_documents)
        return {"output": response}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        # TODO: support only complete for now
        return InputKeys.from_keys({"prompt", "image_documents"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})
