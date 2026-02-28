import logging
from typing import Any, Optional, Sequence, Type

from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
    ValidationError,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import ChatPromptTemplate

logger = logging.getLogger(__name__)


_CORRECTION_PROMPT = (
    "Your previous response could not be parsed into the required structured "
    "format. Please respond ONLY with valid JSON matching the schema, "
    "with no extra text."
)


class StructuredLLM(LLM):
    """
    A structured LLM takes in an inner LLM along with a designated output class,
    and all methods will return outputs in that structure.

    """

    llm: SerializeAsAny[LLM]
    output_cls: Type[BaseModel] = Field(
        ..., description="Output class for the structured LLM.", exclude=True
    )
    max_retries: int = Field(
        default=1,
        description="Maximum number of retries when the LLM fails to produce "
        "valid structured output.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "structured_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return self.llm.metadata

    def _coerce_output(self, output: Any) -> BaseModel:
        """Coerce a potentially non-model output into the expected class."""
        if isinstance(output, self.output_cls):
            return output
        if isinstance(output, str):
            return self.output_cls.model_validate_json(output)
        raise TypeError(
            f"structured_predict returned unexpected type {type(output).__name__}; "
            f"expected {self.output_cls.__name__} or JSON string"
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM."""
        chat_prompt = ChatPromptTemplate(message_templates=messages)

        last_error: Optional[Exception] = None
        for attempt in range(1 + self.max_retries):
            output = self.llm.structured_predict(
                output_cls=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
            )
            try:
                parsed = self._coerce_output(output)
                return ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=parsed.model_dump_json(),
                    ),
                    raw=parsed,
                )
            except (ValidationError, TypeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Structured output attempt %d/%d failed: %s",
                    attempt + 1,
                    1 + self.max_retries,
                    exc,
                )
                # Append a correction hint for the next attempt
                chat_prompt = ChatPromptTemplate(
                    message_templates=[
                        *messages,
                        ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=str(output),
                        ),
                        ChatMessage(
                            role=MessageRole.USER,
                            content=_CORRECTION_PROMPT,
                        ),
                    ]
                )
        raise ValueError(
            f"LLM failed to produce valid {self.output_cls.__name__} output "
            f"after {1 + self.max_retries} attempts"
        ) from last_error

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        chat_prompt = ChatPromptTemplate(message_templates=messages)

        stream_output = self.llm.stream_structured_predict(
            output_cls=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
        )
        for partial_output in stream_output:
            yield ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT, content=partial_output.json()
                ),
                raw=partial_output,
            )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Stream completion endpoint for LLM."""
        raise NotImplementedError("stream_complete is not supported by default.")

    # ===== Async Endpoints =====
    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        chat_prompt = ChatPromptTemplate(message_templates=messages)

        last_error: Optional[Exception] = None
        for attempt in range(1 + self.max_retries):
            output = await self.llm.astructured_predict(
                output_cls=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
            )
            try:
                parsed = self._coerce_output(output)
                return ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=parsed.model_dump_json(),
                    ),
                    raw=parsed,
                )
            except (ValidationError, TypeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Structured output attempt %d/%d failed: %s",
                    attempt + 1,
                    1 + self.max_retries,
                    exc,
                )
                chat_prompt = ChatPromptTemplate(
                    message_templates=[
                        *messages,
                        ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=str(output),
                        ),
                        ChatMessage(
                            role=MessageRole.USER,
                            content=_CORRECTION_PROMPT,
                        ),
                    ]
                )
        raise ValueError(
            f"LLM failed to produce valid {self.output_cls.__name__} output "
            f"after {1 + self.max_retries} attempts"
        ) from last_error

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async stream chat endpoint for LLM."""

        async def gen() -> ChatResponseAsyncGen:
            chat_prompt = ChatPromptTemplate(message_templates=messages)

            stream_output = await self.llm.astream_structured_predict(
                output_cls=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
            )
            async for partial_output in stream_output:
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT, content=partial_output.json()
                    ),
                    raw=partial_output,
                )

        return gen()

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = achat_to_completion_decorator(self.achat)
        return await complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Async stream completion endpoint for LLM."""
        raise NotImplementedError("astream_complete is not supported by default.")
