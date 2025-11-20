from typing import Any, Optional, Sequence, List
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.llm import MessagesToPromptType, CompletionToPromptType
from llama_index.core.types import PydanticProgramMode

from pydantic import Field
import copy


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


class MockLLMWithChatMemoryOfLastCall(MockLLM):
    """
    Mock LLM that keeps track of chat messages of function calls.

    The idea behind this is to be able to easily checks whether the right messages would have been passed to an actual LLM.
    """

    last_chat_messages: Optional[Sequence[ChatMessage]] = Field(
        default=None, exclude=True
    )
    last_called_chat_function: List[str] = Field(default=[], exclude=True)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        r = super().chat(copy.deepcopy(messages), **kwargs)
        self.last_chat_messages = messages
        self.last_called_chat_function.append("chat")
        return r

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        r = super().stream_chat(copy.deepcopy(messages), **kwargs)
        self.last_chat_messages = messages
        self.last_called_chat_function.append("stream_chat")
        return r

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        r = await super().achat(copy.deepcopy(messages), **kwargs)
        self.last_chat_messages = messages
        self.last_called_chat_function.append("achat")
        return r

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        r = await super().astream_chat(copy.deepcopy(messages), **kwargs)
        self.last_chat_messages = messages
        self.last_called_chat_function.append("astream_chat")
        return r

    def reset_memory(self) -> None:
        self.last_chat_messages = None
        self.last_called_chat_function = []

    @classmethod
    def class_name(cls) -> str:
        return "MockLLMWithChatMemoryOfLastCall"
