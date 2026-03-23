from typing import Any, List, Optional, Sequence, Union
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    ImageBlock,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.llm import MessagesToPromptType, CompletionToPromptType
from llama_index.core.multi_modal_llms.base import MultiModalLLM, MultiModalLLMMetadata
from llama_index.core.schema import ImageNode
from llama_index.core.types import PydanticProgramMode


class MockLLM(CustomLLM):
    max_tokens: Optional[int]

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[MessagesToPromptType] = None,
        completion_to_prompt: Optional[CompletionToPromptType] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    ) -> None:
        super().__init__(
            max_tokens=max_tokens,
            callback_manager=callback_manager or CallbackManager([]),
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MockLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(num_output=self.max_tokens or -1)

    def _generate_text(self, length: int) -> str:
        return " ".join(["text" for _ in range(length)])

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response_text = (
            self._generate_text(self.max_tokens) if self.max_tokens else prompt
        )

        return CompletionResponse(
            text=response_text,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def gen_prompt() -> CompletionResponseGen:
            if not prompt:
                yield CompletionResponse(text="", delta="")
                return

            for ch in prompt:
                yield CompletionResponse(
                    text=prompt,
                    delta=ch,
                )

        def gen_response(max_tokens: int) -> CompletionResponseGen:
            for i in range(max_tokens):
                response_text = self._generate_text(i)
                yield CompletionResponse(
                    text=response_text,
                    delta="text ",
                )

        return gen_response(self.max_tokens) if self.max_tokens else gen_prompt()


class MockLLMWithNonyieldingChatStream(MockLLM):
    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        yield from []


class MockLLMWithChatMemoryOfLastCall(MultiModalLLM):
    """
    Mock MultiModalLLM for testing that records the messages from the most
    recent ``chat`` / ``achat`` call.

    Access the captured messages via :attr:`last_messages_received`.
    """

    _last_messages: List[ChatMessage] = PrivateAttr(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "MockLLMWithChatMemoryOfLastCall"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        return MultiModalLLMMetadata(model_name="mock-multi-modal")

    @property
    def last_messages_received(self) -> List[ChatMessage]:
        """Return the messages passed to the most recent chat/achat call."""
        return self._last_messages

    def complete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        return CompletionResponse(text="mock response")

    def stream_complete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseGen:
        yield CompletionResponse(text="mock response", delta="mock response")

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        self._last_messages = list(messages)
        return ChatResponse(
            message=ChatMessage(
                content="mock multi-modal response", role=MessageRole.ASSISTANT
            )
        )

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        self._last_messages = list(messages)
        delta = "mock multi-modal response"
        yield ChatResponse(
            message=ChatMessage(content=delta, role=MessageRole.ASSISTANT),
            delta=delta,
        )

    async def acomplete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        return CompletionResponse(text="mock response")

    async def astream_complete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        async def _gen() -> CompletionResponseAsyncGen:
            yield CompletionResponse(text="mock response", delta="mock response")

        return _gen()

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        self._last_messages = list(messages)
        return ChatResponse(
            message=ChatMessage(
                content="mock multi-modal response", role=MessageRole.ASSISTANT
            )
        )

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        self._last_messages = list(messages)

        async def _gen() -> ChatResponseAsyncGen:
            delta = "mock multi-modal response"
            yield ChatResponse(
                message=ChatMessage(content=delta, role=MessageRole.ASSISTANT),
                delta=delta,
            )

        return _gen()
