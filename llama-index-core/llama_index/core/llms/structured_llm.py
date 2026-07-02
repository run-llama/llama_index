from typing import Any, Sequence, Type

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
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.llm import LLM, StructuredPredictionResult
from llama_index.core.prompts.base import ChatPromptTemplate


class StructuredLLM(LLM):
    """
    A structured LLM takes in an inner LLM along with a designated output class,
    and all methods will return outputs in that structure.

    """

    llm: SerializeAsAny[LLM]
    output_cls: Type[BaseModel] = Field(
        ..., description="Output class for the structured LLM.", exclude=True
    )

    @classmethod
    def class_name(cls) -> str:
        return "structured_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return self.llm.metadata

    def _validate_structured_output(self, output: Any, method_name: str) -> BaseModel:
        if not isinstance(output, BaseModel):
            raise TypeError(
                f"StructuredLLM expected a {self.output_cls.__name__} instance "
                f"from {method_name}, but got {type(output).__name__}: "
                f"{output!r}. The underlying LLM failed to produce valid "
                f"structured output."
            )
        return output

    @staticmethod
    def _prepare_response_kwargs(
        result: StructuredPredictionResult[BaseModel],
    ) -> tuple[Any, dict, Any]:
        source_response = result.source_response
        output = result.output
        raw = output
        additional_kwargs: dict = {}
        logprobs = None

        if source_response is not None:
            raw = source_response.raw if source_response.raw is not None else output
            additional_kwargs = dict(source_response.additional_kwargs or {})
            logprobs = source_response.logprobs

        additional_kwargs.setdefault("structured_output", output)
        return raw, additional_kwargs, logprobs

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM."""
        # NOTE: we are wrapping existing messages in a ChatPromptTemplate to
        # make this work with our FunctionCallingProgram, even though
        # the messages don't technically have any variables (they are already formatted)

        chat_prompt = ChatPromptTemplate(message_templates=messages)

        result = self.llm._structured_predict_with_response(
            output_cls=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
        )
        output = self._validate_structured_output(
            result.output, "_structured_predict_with_response"
        )
        raw, additional_kwargs, logprobs = self._prepare_response_kwargs(result)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=output.model_dump_json()
            ),
            raw=raw,
            logprobs=logprobs,
            additional_kwargs=additional_kwargs,
        )

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
        # NOTE: we are wrapping existing messages in a ChatPromptTemplate to
        # make this work with our FunctionCallingProgram, even though
        # the messages don't technically have any variables (they are already formatted)

        chat_prompt = ChatPromptTemplate(message_templates=messages)

        result = await self.llm._astructured_predict_with_response(
            output_cls=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
        )
        output = self._validate_structured_output(
            result.output, "_astructured_predict_with_response"
        )
        raw, additional_kwargs, logprobs = self._prepare_response_kwargs(result)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=output.model_dump_json()
            ),
            raw=raw,
            logprobs=logprobs,
            additional_kwargs=additional_kwargs,
        )

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
